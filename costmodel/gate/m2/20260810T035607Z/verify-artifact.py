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
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_name))
        requirements = metadata.get_all("Requires-Dist", [])
        assert requirements and all('extra == "' in item for item in requirements)
        for relative in (
            "tilefoundry_costmodel/tileop.py",
            "tilefoundry_costmodel/build.py",
            "tilefoundry_costmodel/implementations/base.py",
            "tilefoundry_costmodel/implementations/registry.py",
            "tilefoundry_costmodel/implementations/synthetic.py",
            "tilefoundry_costmodel/profiler/base.py",
        ):
            assert relative in names
            assert archive.read(relative) == (root / "src" / relative).read_bytes()
        schema_names = sorted(path.name for path in (root / "schemas").glob("*.json"))
        assert len(schema_names) == 7
        for name in schema_names:
            source = (root / "schemas" / name).read_bytes()
            member = next(item for item in names if item.endswith(f"/schemas/{name}"))
            assert archive.read(member) == source
            Draft202012Validator.check_schema(json.loads(source))
        calibration = (root / "calibration" / "b200-hardware.json").read_bytes()
        member = next(item for item in names if item.endswith("/calibration/b200-hardware.json"))
        assert archive.read(member) == calibration
        hardware_schema = json.loads((root / "schemas" / "hardware-v1.schema.json").read_text())
        Draft202012Validator(hardware_schema).validate(json.loads(calibration))
    print(f"wheel_sha256={digest}")
    print("base_dependencies=[]")
    print(f"optional_requirements={len(requirements)}")
    print("m2_sources=source_wheel_bytes_equal")
    print("schemas=7 meta_validation=PASS")
    print("calibration=schema_validation=PASS")


if __name__ == "__main__":
    main()
