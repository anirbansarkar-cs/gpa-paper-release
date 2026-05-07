"""
AlphaGenome K562 oracle client for GPA pipeline.

Connects to the AG oracle server via Unix domain socket and provides the
same `score()` interface as `K562Oracle` so it can be used as a drop-in
replacement for GPA fitness scoring.

Usage:
    client = AlphaGenomeK562Client("/tmp/ag_oracle_12345.sock")
    scores, gc_fracs = client.score(sequences_tensor)  # same as K562Oracle.score()
    client.close()
"""

import socket
import struct
import time

import numpy as np
import torch


class AlphaGenomeK562Client:
    """Client for AG oracle server with K562Oracle-compatible score() API."""

    def __init__(self, socket_path, timeout=300):
        """
        Args:
            socket_path: Path to AG oracle server Unix socket.
            timeout: Socket timeout in seconds.
        """
        self.socket_path = socket_path
        self.timeout = timeout
        self._sock = None
        self._connect()

    def _connect(self):
        """Establish connection to AG oracle server."""
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect(self.socket_path)

    def _reconnect(self):
        """Reconnect if connection was lost."""
        try:
            self._sock.close()
        except Exception:
            pass
        self._connect()

    def _recv_exact(self, nbytes):
        """Receive exactly nbytes from socket."""
        buf = bytearray()
        while len(buf) < nbytes:
            chunk = self._sock.recv(nbytes - len(buf))
            if not chunk:
                raise ConnectionError("AG server connection closed")
            buf.extend(chunk)
        return bytes(buf)

    def ping(self):
        """Check if server is alive. Returns True if responsive."""
        try:
            self._sock.sendall(struct.pack("<i", -1))
            resp = self._recv_exact(4)
            n = struct.unpack("<i", resp)[0]
            return n == 0
        except Exception:
            return False

    def score(self, sequences_tensor, batch_size=None):
        """Score sequences — returns (scores_np, gc_fracs_np).

        Same interface as K562Oracle.score().

        Args:
            sequences_tensor: (N, 200) int tensor {0..3} in diffusion encoding.
            batch_size: Unused (kept for API compatibility).

        Returns:
            (np.array scores, np.array gc_fractions) — both (N,) float.
        """
        if isinstance(sequences_tensor, torch.Tensor):
            indices = sequences_tensor.cpu().numpy()
        else:
            indices = np.asarray(sequences_tensor)

        N, L = indices.shape
        assert L == 200, f"Expected 200bp sequences, got {L}bp"

        # Compute GC locally (C=1, G=2 in diffusion encoding)
        gc_fracs = ((indices == 1) | (indices == 2)).mean(axis=1).astype(np.float32)

        # Send to server
        indices_int16 = indices.astype(np.int16)
        try:
            self._sock.sendall(struct.pack("<i", N))
            self._sock.sendall(indices_int16.tobytes())

            # Receive scores
            scores_bytes = self._recv_exact(N * 8)
            scores = np.frombuffer(scores_bytes, dtype=np.float64).copy()
        except (ConnectionError, BrokenPipeError, socket.timeout) as e:
            print(f"[AG Client] Connection error: {e}, reconnecting...", flush=True)
            self._reconnect()
            # Retry once
            self._sock.sendall(struct.pack("<i", N))
            self._sock.sendall(indices_int16.tobytes())
            scores_bytes = self._recv_exact(N * 8)
            scores = np.frombuffer(scores_bytes, dtype=np.float64).copy()

        return scores, gc_fracs

    def shutdown_server(self):
        """Send shutdown signal to AG server."""
        try:
            self._sock.sendall(struct.pack("<i", 0))
        except Exception:
            pass

    def close(self):
        """Close client connection (does NOT shut down server)."""
        try:
            self._sock.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
