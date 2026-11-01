"""The build-time download of the registered embedding model, and the check of its weights.

The embedder's image runs this once, while it is built, which is the only time its network is
open: it downloads voyage-4-nano at its registered revision (spec section 11) into
``MODEL_DIR``, then fails the build unless ``model.safetensors`` has the registered size and
SHA-256. The server then loads that directory with the network off.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

MODEL_ID = "voyageai/voyage-4-nano"
REVISION = "5fb34e870f49f19e79c94d5e118f5ee730133279"
MODEL_DIR = "/models/voyage-4-nano"
WEIGHTS = "model.safetensors"
WEIGHTS_SIZE = 692_919_112
WEIGHTS_SHA256 = "3dae0c63c81dcab79ac213af331940a1bf2b8a53ec8646be878552890291ad30"


def weights_problem(path: Path, size: int = WEIGHTS_SIZE,
                    sha256: str = WEIGHTS_SHA256) -> str | None:
    """Why the file at ``path`` is not the registered weights, or None when its size and SHA-256
    are the registered ones."""
    if not path.is_file():
        return f"{path} is missing"
    found = path.stat().st_size
    if found != size:
        return f"{path} has {found} bytes, not {size}"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    if digest.hexdigest() != sha256:
        return f"{path} has SHA-256 {digest.hexdigest()}, not {sha256}"
    return None


def main() -> int:
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, revision=REVISION, local_dir=MODEL_DIR)
    problem = weights_problem(Path(MODEL_DIR) / WEIGHTS)
    if problem is not None:
        print(f"fetch_model: {problem}", file=sys.stderr)
        return 1
    print(f"fetch_model: {MODEL_ID} at {REVISION} in {MODEL_DIR}; {WEIGHTS} has the registered "
          f"{WEIGHTS_SIZE} bytes and SHA-256 {WEIGHTS_SHA256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
