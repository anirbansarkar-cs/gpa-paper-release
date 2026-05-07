"""Score LEDIDI-edited sequences with AlphaGenome eval oracle.

Connects to the AG socket server to score all LEDIDI outputs.
Start the server first:
    conda activate alphagenome
    python scripts/k562_mdlm_gpa/ag_oracle_server.py --socket /tmp/ag_oracle_ledidi.sock

Then run this script (in d3_cuda118 env).
"""

import argparse
import os
import struct
import socket
import sys
import time

import numpy as np
import pandas as pd

BASES = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
OUTPUT_DIR = '${GPA_REPO_ROOT}/results/ledidi_comparison'


def connect_ag(socket_path, timeout=600):
    """Connect to AG oracle server, waiting for ready signal."""
    ready_file = socket_path + '.ready'
    print(f"Waiting for AG server at {socket_path}...")
    t0 = time.time()
    while not os.path.exists(ready_file):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"AG server not ready after {timeout}s")
        time.sleep(2)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    print("  Connected to AG server.")
    return sock


def score_batch_ag(sock, indices):
    """Score sequences through AG socket server.

    Args:
        sock: Connected socket.
        indices: (N, 200) int array.

    Returns:
        (N,) float64 scores.
    """
    N = len(indices)
    # Send: int32 N | N*200 int16 indices
    header = struct.pack('<i', N)
    data = indices.astype(np.int16).tobytes()
    sock.sendall(header + data)

    # Receive: N float64 scores
    n_bytes = N * 8
    buf = b''
    while len(buf) < n_bytes:
        chunk = sock.recv(n_bytes - len(buf))
        if not chunk:
            raise ConnectionError("AG server disconnected")
        buf += chunk

    scores = np.frombuffer(buf, dtype=np.float64)
    return scores


def seq_to_indices(seq_str):
    """Convert sequence string to int array."""
    return np.array([BASES[c] for c in seq_str], dtype=np.int64)


def main():
    p = argparse.ArgumentParser(description='Score LEDIDI seqs with AlphaGenome')
    p.add_argument('--socket', required=True, help='AG server socket path')
    p.add_argument('--input_dir', default=OUTPUT_DIR,
                   help='Dir containing ledidi_edited_*.csv')
    p.add_argument('--output', default=os.path.join(OUTPUT_DIR, 'ledidi_ag_scores.csv'))
    p.add_argument('--batch_size', type=int, default=64)
    args = p.parse_args()

    # Load LEDIDI results — find all ledidi_edited_*.csv files
    import glob
    all_dfs = []
    csv_files = sorted(glob.glob(os.path.join(args.input_dir, 'ledidi_edited_*.csv')))
    # Exclude the 'all' combined file
    csv_files = [f for f in csv_files if not f.endswith('_all.csv')]
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        all_dfs.append(df)
        stype = os.path.basename(csv_path).replace('ledidi_edited_', '').replace('.csv', '')
        print(f"Loaded {len(df)} {stype} sequences from {csv_path}")

    if not all_dfs:
        print("No LEDIDI result CSVs found. Run run_ledidi_baseline.py first.")
        sys.exit(1)

    df = pd.concat(all_dfs, ignore_index=True)
    ok_mask = df['status'] == 'ok'
    print(f"Total: {len(df)}, scoring {ok_mask.sum()} successful edits")

    # Convert sequences to indices
    edited_indices = np.array([seq_to_indices(s) for s in df.loc[ok_mask, 'edited_sequence']])
    seed_indices = np.array([seq_to_indices(s) for s in df.loc[ok_mask, 'seed_sequence']])

    # Connect to AG server
    sock = connect_ag(args.socket)

    # Score edited sequences
    print(f"\nScoring {len(edited_indices)} edited sequences...")
    edited_ag = np.zeros(len(edited_indices), dtype=np.float64)
    for start in range(0, len(edited_indices), args.batch_size):
        end = min(start + args.batch_size, len(edited_indices))
        batch = edited_indices[start:end]
        edited_ag[start:end] = score_batch_ag(sock, batch)
        if (start // args.batch_size + 1) % 5 == 0:
            print(f"  Scored {end}/{len(edited_indices)}")

    # Score seed sequences
    print(f"Scoring {len(seed_indices)} seed sequences...")
    seed_ag = np.zeros(len(seed_indices), dtype=np.float64)
    for start in range(0, len(seed_indices), args.batch_size):
        end = min(start + args.batch_size, len(seed_indices))
        batch = seed_indices[start:end]
        seed_ag[start:end] = score_batch_ag(sock, batch)

    # Close connection (don't shutdown server — may be shared across runs)
    sock.close()

    # Add scores to dataframe
    df.loc[ok_mask, 'edited_ag_score'] = edited_ag
    df.loc[ok_mask, 'seed_ag_score'] = seed_ag
    df.loc[ok_mask, 'ag_delta'] = edited_ag - seed_ag

    # Save
    df.to_csv(args.output, index=False)
    print(f"\nSaved AG scores to {args.output}")

    # Summary by seed type
    for stype in df['seed_type'].unique():
        sub = df[(df['seed_type'] == stype) & ok_mask]
        if len(sub) > 0:
            print(f"\n{stype.upper()} seeds:")
            print(f"  LegNet: seed {sub['seed_legnet'].mean():.2f} -> edited {sub['edited_legnet'].mean():.2f}")
            print(f"  AG:     seed {sub['seed_ag_score'].mean():.2f} -> edited {sub['edited_ag_score'].mean():.2f}")
            print(f"  AG delta: {sub['ag_delta'].mean():.2f}")
            print(f"  Exploitation signal: LegNet delta {sub['edited_legnet'].mean() - sub['seed_legnet'].mean():.2f}, "
                  f"AG delta {sub['ag_delta'].mean():.2f}")


if __name__ == '__main__':
    main()
