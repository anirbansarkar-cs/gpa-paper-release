#!/usr/bin/env python3
"""
AlphaGenome K562 oracle server over Unix domain socket.

Runs in the `alphagenome` conda environment (JAX).  Loads the AG K562
checkpoint once at startup and serves score requests from the GPA process
running in the `d3_cuda118` environment (PyTorch).

Protocol (binary, little-endian):
  Request:  4-byte int32 N  |  N*200 int16 indices (row-major)
  Response: N float64 scores

Special messages:
  N = 0   → server shuts down gracefully
  N = -1  → ping (server replies with N=0, zero scores)

Usage:
    python scripts/k562_mdlm_gpa/ag_oracle_server.py \
        --socket /tmp/ag_oracle_12345.sock \
        --batch_size 64
"""

import argparse
import os
import signal
import socket
import struct
import sys
import time

import numpy as np


def load_ag_oracle(batch_size=64):
    """Load AlphaGenome K562 oracle (JAX).

    Old path ${GPA_SHARED_ROOT}/alphagenome_encoder/ was deleted; the
    JAX ckpts are now under ${GPA_SHARED_ROOT}/models/alphagenome_encoder/jax/
    and need a runtime config patch (use_encoder_output flag) — both handled
    by scripts/alphagenome/score_sequences_jax.load_jax_oracle.
    """
    sys.path.insert(0, "${GPA_REPO_ROOT}")
    from scripts.alphagenome.score_sequences_jax import load_jax_oracle

    cell = os.environ.get("AG_CELL", "k562").lower()
    stage = os.environ.get("AG_STAGE", "stage1").lower()
    print(f"[AG Server] Loading {cell} JAX oracle ({stage}) "
          f"via score_sequences_jax", flush=True)
    t0 = time.time()
    oracle = load_jax_oracle(cell, stage=stage)
    print(f"[AG Server] Loaded in {time.time() - t0:.1f}s", flush=True)
    return oracle


def indices_to_onehot(indices):
    """Convert (N, L) int indices to (N, L, 4) one-hot float32."""
    N, L = indices.shape
    onehot = np.zeros((N, L, 4), dtype=np.float32)
    for i in range(4):
        onehot[:, :, i] = (indices == i).astype(np.float32)
    return onehot


def score_batch(oracle, indices, batch_size=64):
    """Score indices (N, 200) with AG oracle, returning RC-AVERAGED scores.

    Per LegNet-paper TTA convention (and confirmed for AG empirically — fwd vs
    RC differs by ~0.1-0.2 per seq, Pearson 0.93), averaging fwd+RC inside the
    optimization loop ensures pool selection AND ARCHIVE_THRESHOLD
    decisions both use the honest, strand-invariant metric — not just fwd-only.
    Cost: 2x AG inference per call. Worth it for honesty.

    RC of a (N, 200, 4) onehot in A=0,C=1,G=2,T=3 channel order:
      reverse positions, swap A<->T (channels 0<->3) and C<->G (channels 1<->2).
    """
    onehot = indices_to_onehot(indices)                         # (N, 200, 4)
    onehot_rc = onehot[:, ::-1, :].copy()[:, :, [3, 2, 1, 0]]   # RC
    N = onehot.shape[0]
    fwd_scores, rc_scores = [], []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        f = oracle.predict(onehot[start:end], mode="core", batch_size=batch_size)
        r = oracle.predict(onehot_rc[start:end], mode="core", batch_size=batch_size)
        fwd_scores.append(np.asarray(f, dtype=np.float64))
        rc_scores.append(np.asarray(r, dtype=np.float64))
    fwd = np.concatenate(fwd_scores, axis=0)
    rc  = np.concatenate(rc_scores, axis=0)
    return (fwd + rc) / 2.0


def recv_exact(conn, nbytes):
    """Receive exactly nbytes from socket."""
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = conn.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed")
        buf.extend(chunk)
    return bytes(buf)


def serve(socket_path, batch_size=64):
    """Main server loop."""
    oracle = load_ag_oracle(batch_size)

    # Clean up stale socket
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(socket_path)
    sock.listen(1)
    print(f"[AG Server] Listening on {socket_path}", flush=True)

    # Write ready signal
    ready_path = socket_path + ".ready"
    with open(ready_path, "w") as f:
        f.write("ready\n")

    def cleanup(signum=None, frame=None):
        print("[AG Server] Shutting down", flush=True)
        sock.close()
        try:
            os.unlink(socket_path)
        except OSError:
            pass
        try:
            os.unlink(ready_path)
        except OSError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    try:
        while True:
            conn, _ = sock.accept()
            try:
                while True:
                    # Read N (int32)
                    header = recv_exact(conn, 4)
                    N = struct.unpack("<i", header)[0]

                    if N == 0:
                        # Shutdown signal
                        print("[AG Server] Received shutdown signal", flush=True)
                        conn.close()
                        cleanup()
                        return

                    if N == -1:
                        # Ping
                        conn.sendall(struct.pack("<i", 0))
                        continue

                    # Read indices: N * 200 * 2 bytes (int16)
                    seq_len = 200
                    data_bytes = N * seq_len * 2
                    raw = recv_exact(conn, data_bytes)
                    indices = np.frombuffer(raw, dtype=np.int16).reshape(N, seq_len).astype(np.int64)

                    # Score
                    t0 = time.time()
                    scores = score_batch(oracle, indices, batch_size)
                    elapsed = time.time() - t0
                    print(f"[AG Server] Scored {N} seqs in {elapsed:.2f}s  "
                          f"(mean={scores.mean():.4f}, max={scores.max():.4f})",
                          flush=True)

                    # Send scores: N * 8 bytes (float64)
                    conn.sendall(scores.tobytes())

            except ConnectionError:
                print("[AG Server] Client disconnected", flush=True)
            finally:
                conn.close()
    except Exception as e:
        print(f"[AG Server] Error: {e}", flush=True)
    finally:
        cleanup()


def main():
    parser = argparse.ArgumentParser(description="AlphaGenome K562 oracle server")
    parser.add_argument("--socket", required=True,
                        help="Unix domain socket path")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="AG inference batch size")
    args = parser.parse_args()
    serve(args.socket, args.batch_size)


if __name__ == "__main__":
    main()
