from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

from jsonschema import Draft202012Validator


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: verify-artifact.py WHEEL COSTMODEL_ROOT M0_WHEEL")
    wheel = Path(sys.argv[1])
    root = Path(sys.argv[2])
    m0_wheel = Path(sys.argv[3])
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_name))
        requirements = metadata.get_all("Requires-Dist", [])
        assert requirements
        assert all('extra == "' in requirement for requirement in requirements)

        schema_names = sorted(path.name for path in (root / "schemas").glob("*.json"))
        assert len(schema_names) == 7
        for name in schema_names:
            source = (root / "schemas" / name).read_bytes()
            member = next(member for member in names if member.endswith(f"/schemas/{name}"))
            assert archive.read(member) == source
            Draft202012Validator.check_schema(json.loads(source))

        calibration_name = "b200-hardware.json"
        calibration = (root / "calibration" / calibration_name).read_bytes()
        member = next(
            member for member in names if member.endswith(f"/calibration/{calibration_name}")
        )
        assert archive.read(member) == calibration
        hardware_schema = json.loads((root / "schemas" / "hardware-v1.schema.json").read_text())
        Draft202012Validator(hardware_schema).validate(json.loads(calibration))

        root_init = archive.read("tilefoundry_costmodel/__init__.py")
    with zipfile.ZipFile(m0_wheel) as archive:
        assert archive.read("tilefoundry_costmodel/__init__.py") == root_init

    print(f"wheel_sha256={digest}")
    print("base_dependencies=[]")
    print(f"optional_requirements={len(requirements)}")
    print("schemas=7 source_wheel_bytes=equal meta_validation=PASS")
    print("calibration=source_wheel_bytes=equal schema_validation=PASS")
    print("root_boundary=M0_bytes_equal")


if __name__ == "__main__":
    main()
