"""Download the pinned Tiny Shakespeare text and verify its SHA-256."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
       "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt")
SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def main():
    path = Path("data/input.txt")
    data = path.read_bytes() if path.exists() else urlopen(URL, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("dataset checksum mismatch; existing files are not overwritten")
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(data)
    print(f"Verified {path}: {len(data):,} bytes")


if __name__ == "__main__":
    main()
