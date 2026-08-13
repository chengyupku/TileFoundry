from __future__ import annotations

import hashlib
import sys
from pathlib import Path

EXPECTED_SHA256 = "1acaec7311fd9a1cb4795bf551693f4eaac5bdd333f06a146ae7007725bb72f7"


def payload(path: Path) -> bytes:
    for line in path.read_bytes().splitlines():
        if line.startswith(b"[{"):
            return line
    raise AssertionError(f"no Golden payload in {path}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: verify-golden.py LOG ...")
    paths = tuple(Path(name) for name in sys.argv[1:])
    payloads = tuple(payload(path) for path in paths)
    assert len(set(payloads)) == 1
    digest = hashlib.sha256(payloads[0]).hexdigest()
    assert digest == EXPECTED_SHA256
    print(f"golden_sha256={digest}")
    print(f"environments={len(paths)}")
    print("source_wheel_golden_bytes=identical")


if __name__ == "__main__":
    main()
