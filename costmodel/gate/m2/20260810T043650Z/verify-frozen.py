from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

ROOT_SHA256 = "e7f7fd0ed8db95273f63ac1f987cdc3d182418f504e3a6c19a19d567c5d09ff7"
M0_WHEEL_SHA256 = "57058acb41622f7119da2389d96c9345df0523f5f355a83774a9559d9b2e949a"
M1_WHEEL_SHA256 = "fbe0f7619b919ce2442b81051a36f84c9bc091d8d0f874475d53f94b21f4a84a"
M2_WHEEL_SHA256 = "d69725f1f0afdea1b08e810faaa98e3a0631eb2846f7a5dcf32cdbd31849e3f8"
M0_MANIFEST_SHA256 = "a7bc6bbf4347db9f7f63f1624e500e35d08596730989a208784ba125eb1652e1"
M1_MANIFEST_SHA256 = "1fba738b2488c0e05f635372c03dd877565d47828edd925542daf430356f1ed9"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wheel_root_digest(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        return hashlib.sha256(archive.read("tilefoundry_costmodel/__init__.py")).hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify-frozen.py REPOSITORY_ROOT")
    root = Path(sys.argv[1]).resolve()
    m0 = root / "costmodel/gate/m0/20260807T041107Z"
    m1 = root / "costmodel/gate/m1/20260807T092506Z"
    m2 = root / "costmodel/gate/m2/20260810T043650Z"
    wheels = (
        ("m0", m0 / "wheel/tilefoundry_costmodel-2.0.0-py3-none-any.whl"),
        ("m1", m1 / "wheel/tilefoundry_costmodel-2.0.0-py3-none-any.whl"),
        ("m2", m2 / "wheel/tilefoundry_costmodel-2.0.0-py3-none-any.whl"),
    )
    assert digest(wheels[0][1]) == M0_WHEEL_SHA256
    assert digest(wheels[1][1]) == M1_WHEEL_SHA256
    assert digest(wheels[2][1]) == M2_WHEEL_SHA256
    assert digest(m0 / "SHA256SUMS.txt") == M0_MANIFEST_SHA256
    assert digest(m1 / "SHA256SUMS.txt") == M1_MANIFEST_SHA256
    source_root = digest(root / "costmodel/src/tilefoundry_costmodel/__init__.py")
    assert source_root == ROOT_SHA256
    for name, wheel in wheels:
        archive_root = wheel_root_digest(wheel)
        assert archive_root == ROOT_SHA256
        print(f"{name}_wheel_sha256={digest(wheel)}")
        print(f"{name}_root_sha256={archive_root}")
    print(f"source_root_sha256={source_root}")
    print(f"m0_manifest_sha256={digest(m0 / 'SHA256SUMS.txt')}")
    print(f"m1_manifest_sha256={digest(m1 / 'SHA256SUMS.txt')}")
    print("frozen_boundary=PASS")


if __name__ == "__main__":
    main()
