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
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify-artifact.py WHEEL COSTMODEL_ROOT")
    wheel = Path(sys.argv[1])
    root = Path(sys.argv[2])
    wheel_digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        package_members = sorted(
            name
            for name in names
            if name.startswith("tilefoundry_costmodel/")
            and name.endswith((".py", ".cu"))
            and "/__pycache__/" not in name
        )
        source_root = root / "src"
        for member in package_members:
            source = source_root / member
            assert source.is_file(), member
            assert archive.read(member) == source.read_bytes(), member

        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_name))
        requirements = metadata.get_all("Requires-Dist", [])
        assert not [item for item in requirements if 'extra == "' not in item], requirements
        assert {"cuda", "test"} <= set(metadata.get_all("Provides-Extra", []))

        schema_names = sorted(path.name for path in (root / "schemas").glob("*.json"))
        assert len(schema_names) == 7
        for schema_name in schema_names:
            source = (root / "schemas" / schema_name).read_bytes()
            member = next(item for item in names if item.endswith(f"/schemas/{schema_name}"))
            assert archive.read(member) == source
            Draft202012Validator.check_schema(json.loads(source))

        calibration = (root / "calibration" / "b200-hardware.json").read_bytes()
        member = next(item for item in names if item.endswith("/calibration/b200-hardware.json"))
        assert archive.read(member) == calibration
        hardware_schema = json.loads((root / "schemas" / "hardware-v1.schema.json").read_text())
        Draft202012Validator(hardware_schema).validate(json.loads(calibration))

    print(f"wheel_sha256={wheel_digest}")
    print(f"package_source_members={len(package_members)}")
    print("source_wheel_parity=PASS")
    print("base_dependencies=[]")
    print("optional_extras=cuda,test")
    print("schemas=7 meta_validation=PASS")
    print("calibration=schema_validation=PASS")


if __name__ == "__main__":
    main()
