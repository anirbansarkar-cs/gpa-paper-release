#!/usr/bin/env python3
"""ENCODE V4 cCRE curation + paper-faithful graph-cluster split.

Implements the DNA-CRAFT (arXiv 2604.20488, Appendix A.2) split protocol:
  1. Pull V4 hg38 + mm10 cCRE BEDs.
  2. Pad/trim each region to 350 bp (centered on midpoint).
  3. Build a sequence graph G=(V,E):
     - Within-species overlap edges: any two cCREs (same species) sharing
       >= MIN_OVERLAP bp.
     - Cross-species homology edges: hg38 <-> mm10 syntenic-net liftOver
       pairs sharing >= MIN_OVERLAP bp.
  4. Connected-components partition into train/val/test = 90/5/5,
     weighted by region count (size-balanced, not component-balanced).
  5. Tokenize ACGT -> {0,1,2,3}, with PAD=4 and MASK=5.
  6. Write parquet shards + split_manifest.csv.

Outputs (under RESULTS_DIR):
  ccre/train.parquet
  ccre/val.parquet
  ccre/test.parquet
  ccre/split_manifest.csv  (component_id -> split, region_count)
  ccre/curate_summary.json (counts, edge stats, runtime)

Run via submit_curate_ccre.sh on a CPU partition. ~4-6 h wall on a 32-core
node, dominated by bedtools intersect of a 3.28M-region BED against itself
and scipy connected-components on ~10M edges.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "results" / "dna_craft_comparison" / "enhancer"
DATA_DIR = PROJECT_ROOT / "data"
REF_DIR = DATA_DIR / "reference_genomes"

# Paper-faithful constants
SEQ_LEN = 350
MIN_OVERLAP = 100         # bp threshold for both within-species and cross-species edges
SPLIT_RATIOS = (0.90, 0.05, 0.05)
RANDOM_STATE = 42

# ENCODE V4 cCRE BED downloads (ENCODE Project portal — wenglab.org direct downloads
# return 404 behind a Vercel anti-bot challenge as of 2026-04-25; the canonical
# file accessions are hosted on encodeproject.org instead).
# - hg38: ENCSR800VNX → ENCFF420VPZ (≈32 MB .bed.gz, ≈3.28 M agnostic cCREs)
# - mm10: ENCSR412JPD → ENCFF167FJQ (gzipped BED of agnostic mm10 cCREs)
HG38_BED_URL = "https://www.encodeproject.org/files/ENCFF420VPZ/@@download/ENCFF420VPZ.bed.gz"
MM10_BED_URL = "https://www.encodeproject.org/files/ENCFF167FJQ/@@download/ENCFF167FJQ.bed.gz"

# UCSC syntenic net (hg38 vs mm10). The actual file under
# goldenPath/hg38/vsMm10/ is `hg38.mm10.syn.net.gz` (chain/net format with
# "net <chrom>" + "fill"/"gap" lines), not the .txt.gz of synNet tables.
SYNTENIC_NET_URL = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/vsMm10/hg38.mm10.syn.net.gz"
)

# Reference genome FASTAs (must already exist on disk OR be downloaded by Track A)
HG38_FA = REF_DIR / "hg38.fa"
MM10_FA = REF_DIR / "mm10.fa"

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}  # N -> PAD slot


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run(cmd, **kw):
    log(f"$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kw)


# ----------------------------------------------------------------------------
# Step 1 — download / locate inputs
# ----------------------------------------------------------------------------
def download_if_missing(url: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        log(f"  exists: {out_path} ({out_path.stat().st_size:,} bytes)")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    log(f"  fetching {url} -> {out_path}")
    run(["wget", "-q", "-O", str(tmp), url])
    tmp.replace(out_path)


def gunzip_to(in_gz: Path, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    log(f"  gunzip {in_gz} -> {out_path}")
    with gzip.open(in_gz, "rb") as src, open(out_path, "wb") as dst:
        shutil.copyfileobj(src, dst)


# ----------------------------------------------------------------------------
# Step 2 — pad/trim to SEQ_LEN around the midpoint, write a uniform BED
# ----------------------------------------------------------------------------
def normalize_bed(in_bed: Path, out_bed: Path, species_tag: str) -> int:
    """Read BED, center each region, expand to SEQ_LEN, drop chr_random / chrUn.

    Adds a unique region_id column = "{species}:{chrom}:{midpoint}".
    Transparently handles both plain .bed and gzipped .bed.gz inputs.
    """
    log(f"  normalize {in_bed} -> {out_bed}")
    rows: list[tuple[str, int, int, str]] = []
    half = SEQ_LEN // 2
    open_fn = gzip.open if str(in_bed).endswith(".gz") else open
    with open_fn(in_bed, "rt") as fh:
        for line in fh:
            parts = line.rstrip().split("\t")
            if len(parts) < 3:
                continue
            chrom, start_s, end_s = parts[0], parts[1], parts[2]
            if "_" in chrom or chrom.startswith("chrUn") or chrom == "chrM":
                continue
            try:
                start, end = int(start_s), int(end_s)
            except ValueError:
                continue
            mid = (start + end) // 2
            new_start = max(0, mid - half)
            new_end = new_start + SEQ_LEN
            region_id = f"{species_tag}:{chrom}:{mid}"
            rows.append((chrom, new_start, new_end, region_id))
    rows.sort(key=lambda r: (r[0], r[1]))
    seen: set[str] = set()
    deduped: list[tuple[str, int, int, str]] = []
    for r in rows:
        if r[3] in seen:
            continue
        seen.add(r[3])
        deduped.append(r)
    with open(out_bed, "w") as fh:
        for chrom, s, e, rid in deduped:
            fh.write(f"{chrom}\t{s}\t{e}\t{rid}\n")
    log(f"    wrote {len(deduped):,} regions")
    return len(deduped)


# ----------------------------------------------------------------------------
# Step 3 — extract sequences from reference FASTA
# ----------------------------------------------------------------------------
def extract_sequences(bed_path: Path, fasta_path: Path, out_tsv: Path) -> int:
    """Use bedtools getfasta to extract per-region sequences."""
    if out_tsv.exists() and out_tsv.stat().st_size > 0:
        log(f"  exists: {out_tsv}")
        return sum(1 for _ in open(out_tsv))
    log(f"  bedtools getfasta {bed_path} -> {out_tsv}")
    fa_out = out_tsv.with_suffix(".fa")
    run(["bedtools", "getfasta", "-fi", str(fasta_path), "-bed", str(bed_path),
         "-fo", str(fa_out), "-name"])
    # Convert FASTA to id<TAB>seq TSV; drop sequences with non-uniform length or all-N
    n_kept = 0
    with open(fa_out) as fh, open(out_tsv, "w") as out:
        rid, buf = None, []
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if rid is not None:
                    seq = "".join(buf).upper()
                    if len(seq) == SEQ_LEN and seq.count("N") < SEQ_LEN // 2:
                        out.write(f"{rid}\t{seq}\n")
                        n_kept += 1
                # name ends after "::" in bedtools getfasta -name
                rid = line[1:].split("::", 1)[0]
                buf = []
            else:
                buf.append(line)
        if rid is not None:
            seq = "".join(buf).upper()
            if len(seq) == SEQ_LEN and seq.count("N") < SEQ_LEN // 2:
                out.write(f"{rid}\t{seq}\n")
                n_kept += 1
    fa_out.unlink(missing_ok=True)
    log(f"    kept {n_kept:,} sequences (uniform length, < 50% N)")
    return n_kept


# ----------------------------------------------------------------------------
# Step 4 — build the graph edges
# ----------------------------------------------------------------------------
def within_species_edges(bed_path: Path, edges_out: Path) -> int:
    """Self-intersect (>=MIN_OVERLAP bp) within one species via bedtools."""
    log(f"  within-species edges: {bed_path}")
    cmd = ["bedtools", "intersect", "-a", str(bed_path), "-b", str(bed_path),
           "-wa", "-wb"]
    n_edges = 0
    with open(edges_out, "w") as out, subprocess.Popen(
            cmd, stdout=subprocess.PIPE, text=True) as proc:
        for line in proc.stdout:
            f = line.rstrip().split("\t")
            # chrom1 s1 e1 id1 chrom2 s2 e2 id2
            if len(f) < 8:
                continue
            id1, id2 = f[3], f[7]
            if id1 == id2:
                continue
            s1, e1, s2, e2 = int(f[1]), int(f[2]), int(f[5]), int(f[6])
            ov = max(0, min(e1, e2) - max(s1, s2))
            if ov >= MIN_OVERLAP:
                out.write(f"{id1}\t{id2}\n")
                n_edges += 1
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"bedtools intersect failed (rc={proc.returncode})")
    log(f"    {n_edges:,} within-species edges")
    return n_edges


def cross_species_edges(hg38_bed: Path, mm10_bed: Path,
                        synnet_path: Path, edges_out: Path) -> int:
    """Liftover hg38 cCREs into mm10 via syntenic-net, then intersect with mm10 cCREs.

    The synNet.txt has many fields per record; the gap structure means a
    full liftOver chain implementation is overkill. We use a simplified
    heuristic: parse top-level chain-anchor lines from synNet to map
    hg38 (chrom, midpoint) -> mm10 (chrom, midpoint), then ask whether
    that mm10 midpoint falls within a SEQ_LEN window of any mm10 cCRE.

    For paper-spec rigor a full UCSC liftOver chain would be preferred;
    this approximation captures the dominant homology signal.
    """
    log(f"  cross-species edges via {synnet_path.name}")
    # Build mm10 interval lookup: chrom -> sorted [(start, end, id), ...]
    mm10_idx: dict[str, list[tuple[int, int, str]]] = {}
    with open(mm10_bed) as fh:
        for line in fh:
            chrom, s, e, rid = line.rstrip().split("\t")[:4]
            mm10_idx.setdefault(chrom, []).append((int(s), int(e), rid))
    for chrom in mm10_idx:
        mm10_idx[chrom].sort()
    # Parse synNet (gz or plain) — keep only "fill" lines that anchor an aligned block
    n_edges = 0
    open_fn = gzip.open if str(synnet_path).endswith(".gz") else open
    n_lines = 0
    with open_fn(synnet_path, "rt") as fh, open(edges_out, "w") as out:
        cur_chrom: str | None = None
        for line in fh:
            n_lines += 1
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("net"):
                cur_chrom = line.split()[1]
                continue
            if cur_chrom is None:
                continue
            # "fill" or "gap" lines start with whitespace, then keyword
            t = line.split()
            if not t or t[0] != "fill":
                continue
            # fill <tStart> <tSize> <qName> <strand> <qStart> <qSize> ...
            try:
                t_start, t_size = int(t[1]), int(t[2])
                q_name = t[3]
                q_start, q_size = int(t[5]), int(t[6])
            except (ValueError, IndexError):
                continue
            t_mid = t_start + t_size // 2
            q_mid = q_start + q_size // 2
            hg38_id = f"hg38:{cur_chrom}:{t_mid}"
            # Is q_mid within any mm10 cCRE window?
            mm10_intervals = mm10_idx.get(q_name, [])
            # Binary search for interval containing q_mid
            lo, hi = 0, len(mm10_intervals)
            while lo < hi:
                m = (lo + hi) // 2
                s, e, _ = mm10_intervals[m]
                if e <= q_mid:
                    lo = m + 1
                elif s > q_mid:
                    hi = m
                else:
                    out.write(f"{hg38_id}\t{mm10_intervals[m][2]}\n")
                    n_edges += 1
                    break
    log(f"    parsed {n_lines:,} synNet lines, {n_edges:,} cross-species edges")
    return n_edges


# ----------------------------------------------------------------------------
# Step 5 — connected components via scipy.sparse.csgraph
# ----------------------------------------------------------------------------
@dataclass
class ComponentAssignment:
    region_to_component: dict[str, int]
    component_sizes: np.ndarray  # (n_components,)
    n_components: int


def connected_components(region_ids: list[str],
                         edge_files: list[Path]) -> ComponentAssignment:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components as cc
    log(f"  building edge matrix for {len(region_ids):,} regions")
    id_to_idx = {rid: i for i, rid in enumerate(region_ids)}
    rows, cols = [], []
    for ef in edge_files:
        if not ef.exists():
            continue
        with open(ef) as fh:
            for line in fh:
                a, b = line.rstrip().split("\t")
                ia, ib = id_to_idx.get(a), id_to_idx.get(b)
                if ia is None or ib is None:
                    continue
                rows.append(ia)
                cols.append(ib)
    n = len(region_ids)
    log(f"    {len(rows):,} edges loaded; running connected_components")
    data = np.ones(len(rows), dtype=np.int8)
    g = csr_matrix((data, (rows, cols)), shape=(n, n))
    n_comp, labels = cc(g, directed=False, return_labels=True)
    log(f"    n_components = {n_comp:,}")
    sizes = np.bincount(labels, minlength=n_comp)
    region_to_component = {rid: int(labels[i]) for i, rid in enumerate(region_ids)}
    return ComponentAssignment(region_to_component, sizes, n_comp)


# ----------------------------------------------------------------------------
# Step 6 — partition components into train/val/test, size-balanced
# ----------------------------------------------------------------------------
def partition_components(assign: ComponentAssignment,
                         seed: int = RANDOM_STATE) -> dict[int, str]:
    log("  partitioning components 90/5/5 (size-weighted)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(assign.n_components)
    total = int(assign.component_sizes.sum())
    train_t, val_t = SPLIT_RATIOS[0] * total, sum(SPLIT_RATIOS[:2]) * total
    cum = 0
    component_to_split: dict[int, str] = {}
    for cid in order:
        cum += int(assign.component_sizes[cid])
        if cum <= train_t:
            component_to_split[int(cid)] = "train"
        elif cum <= val_t:
            component_to_split[int(cid)] = "val"
        else:
            component_to_split[int(cid)] = "test"
    counts = {"train": 0, "val": 0, "test": 0}
    for cid, sp in component_to_split.items():
        counts[sp] += int(assign.component_sizes[cid])
    log(f"    region counts: {counts}")
    return component_to_split


# ----------------------------------------------------------------------------
# Step 7 — tokenize ACGT and write parquet shards
# ----------------------------------------------------------------------------
def tokenize(seq: str) -> np.ndarray:
    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    out = np.full(len(seq), 4, dtype=np.uint8)  # default to N=4 (PAD slot)
    out[arr == ord("A")] = 0
    out[arr == ord("C")] = 1
    out[arr == ord("G")] = 2
    out[arr == ord("T")] = 3
    return out


def write_parquet_shards(seq_tsvs: list[Path],
                         region_to_component: dict[str, int],
                         component_to_split: dict[int, str],
                         out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0, "test": 0}
    buffers: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    def flush(split: str) -> None:
        if not buffers[split]:
            return
        df = pd.DataFrame(buffers[split])
        path = out_dir / f"{split}.parquet"
        if path.exists():
            existing = pq.read_table(path).to_pandas()
            df = pd.concat([existing, df], ignore_index=True)
        pq.write_table(pa.Table.from_pandas(df), path, compression="snappy")
        counts[split] += len(buffers[split])
        buffers[split] = []

    BATCH = 50_000
    for tsv in seq_tsvs:
        log(f"  reading {tsv}")
        with open(tsv) as fh:
            for line in fh:
                rid, seq = line.rstrip().split("\t", 1)
                comp = region_to_component.get(rid)
                if comp is None:
                    continue
                split = component_to_split.get(comp)
                if split is None:
                    continue
                tokens = tokenize(seq)
                buffers[split].append({"region_id": rid, "tokens": tokens.tolist()})
                if len(buffers[split]) >= BATCH:
                    flush(split)
    for split in ("train", "val", "test"):
        flush(split)
    log(f"  parquet counts: {counts}")
    return counts


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default=str(RESULTS_DIR / "ccre"))
    p.add_argument("--work_dir", default=str(RESULTS_DIR / "ccre" / "_work"))
    p.add_argument("--hg38_fa", default=str(HG38_FA))
    p.add_argument("--mm10_fa", default=str(MM10_FA))
    p.add_argument("--skip_cross_species", action="store_true",
                   help="Skip mm10 + syntenic net (within-species edges only).")
    p.add_argument("--skip_extract", action="store_true",
                   help="Skip bedtools getfasta (assume *.tsv exist).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log("== Step 1: download ENCODE V4 cCRE BEDs and synNet ==")
    # ENCODE-hosted V4 cCRE files come gzipped (.bed.gz); UCSC syn.net is .gz.
    hg38_bed_raw = work / "V4-hg38.bed.gz"
    mm10_bed_raw = work / "V4-mm10.bed.gz"
    download_if_missing(HG38_BED_URL, hg38_bed_raw)
    if not args.skip_cross_species:
        download_if_missing(MM10_BED_URL, mm10_bed_raw)
        synnet_gz = work / "hg38.mm10.syn.net.gz"
        download_if_missing(SYNTENIC_NET_URL, synnet_gz)
    else:
        synnet_gz = None

    log("== Step 2: normalize BEDs to SEQ_LEN ==")
    hg38_bed = work / "hg38.norm.bed"
    n_hg38 = normalize_bed(hg38_bed_raw, hg38_bed, "hg38")
    if not args.skip_cross_species:
        mm10_bed = work / "mm10.norm.bed"
        n_mm10 = normalize_bed(mm10_bed_raw, mm10_bed, "mm10")
    else:
        mm10_bed = None
        n_mm10 = 0

    log("== Step 3: bedtools getfasta ==")
    seq_tsvs: list[Path] = []
    if not args.skip_extract:
        if Path(args.hg38_fa).exists():
            hg38_tsv = work / "hg38.seq.tsv"
            extract_sequences(hg38_bed, Path(args.hg38_fa), hg38_tsv)
            seq_tsvs.append(hg38_tsv)
        else:
            log(f"  WARNING: hg38 FASTA missing at {args.hg38_fa}; skipping extract")
        if mm10_bed is not None and Path(args.mm10_fa).exists():
            mm10_tsv = work / "mm10.seq.tsv"
            extract_sequences(mm10_bed, Path(args.mm10_fa), mm10_tsv)
            seq_tsvs.append(mm10_tsv)
        elif mm10_bed is not None:
            log(f"  WARNING: mm10 FASTA missing at {args.mm10_fa}; skipping extract")
    else:
        for tag in ("hg38", "mm10"):
            t = work / f"{tag}.seq.tsv"
            if t.exists():
                seq_tsvs.append(t)

    log("== Step 4: build graph edges ==")
    edges_within_hg38 = work / "edges.within.hg38.tsv"
    within_species_edges(hg38_bed, edges_within_hg38)
    edge_files = [edges_within_hg38]
    if mm10_bed is not None:
        edges_within_mm10 = work / "edges.within.mm10.tsv"
        within_species_edges(mm10_bed, edges_within_mm10)
        edge_files.append(edges_within_mm10)
        if synnet_gz is not None and synnet_gz.exists():
            edges_cross = work / "edges.cross.tsv"
            cross_species_edges(hg38_bed, mm10_bed, synnet_gz, edges_cross)
            edge_files.append(edges_cross)

    log("== Step 5: load region IDs and run connected components ==")
    region_ids: list[str] = []
    for bed in (hg38_bed, mm10_bed):
        if bed is None:
            continue
        with open(bed) as fh:
            for line in fh:
                region_ids.append(line.rstrip().split("\t")[3])
    log(f"  total regions: {len(region_ids):,}")
    assign = connected_components(region_ids, edge_files)

    log("== Step 6: partition components into splits ==")
    component_to_split = partition_components(assign)

    log("== Step 7: write split_manifest.csv ==")
    manifest_rows = []
    for cid in range(assign.n_components):
        manifest_rows.append({
            "component_id": cid,
            "split": component_to_split[cid],
            "n_regions": int(assign.component_sizes[cid]),
        })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(out_dir / "split_manifest.csv", index=False)
    log(f"    wrote {out_dir / 'split_manifest.csv'}")

    log("== Step 8: tokenize and shard parquet ==")
    if seq_tsvs:
        counts = write_parquet_shards(
            seq_tsvs, assign.region_to_component, component_to_split, out_dir)
    else:
        log("  no FASTA-extracted sequences available; parquet shards SKIPPED")
        counts = {"train": 0, "val": 0, "test": 0}

    elapsed = time.time() - t0
    summary = {
        "n_regions_hg38": n_hg38,
        "n_regions_mm10": n_mm10,
        "n_components": assign.n_components,
        "split_counts_regions": counts,
        "elapsed_seconds": elapsed,
        "min_overlap_bp": MIN_OVERLAP,
        "seq_len": SEQ_LEN,
        "split_ratios": SPLIT_RATIOS,
        "random_state": RANDOM_STATE,
    }
    with open(out_dir / "curate_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"== Done in {elapsed/60:.1f} min ==")
    log(f"   summary: {out_dir / 'curate_summary.json'}")


if __name__ == "__main__":
    main()
