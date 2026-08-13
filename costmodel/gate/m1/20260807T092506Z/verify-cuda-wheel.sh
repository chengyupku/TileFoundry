#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PYTHON WHEEL VENV" >&2
  exit 2
fi

python_bin=$1
wheel=$2
venv=$3

export PIP_DISABLE_PIP_VERSION_CHECK=1

"${python_bin}" -m venv "${venv}"
"${venv}/bin/python" --version
"${venv}/bin/python" -m pip install "${wheel}[cuda]"
"${venv}/bin/python" -m pip check
"${venv}/bin/python" -m pip freeze --all
"${venv}/bin/python" - <<'PY'
import importlib.metadata
import sys

preloaded = sorted(name for name in sys.modules if name == "cuda" or name.startswith("cuda."))
before = set(sys.modules)
import tilefoundry_costmodel as cm

added = set(sys.modules) - before
forbidden = sorted(
    name
    for name in added
    if name == "cuda"
    or name.startswith("cuda.")
    or name == "ortools"
    or name.startswith("ortools.")
    or name == "tilefoundry_costmodel.legacy"
    or name.startswith("tilefoundry_costmodel.legacy.")
)
assert not forbidden, forbidden
assert cm.__file__ is not None and "site-packages" in cm.__file__, cm.__file__
print(f"root={cm.__file__}")
print(f"preloaded_cuda_modules={preloaded}")
print("new_optional_or_legacy_modules=[]")

from cuda.bindings import driver

status = driver.cuInit(0)
assert status == (driver.CUresult.CUDA_SUCCESS,), status
status, count = driver.cuDeviceGetCount()
assert status == driver.CUresult.CUDA_SUCCESS and count > 0, (status, count)
print(f"device_count={count}")
for index in range(count):
    status, device = driver.cuDeviceGet(index)
    assert status == driver.CUresult.CUDA_SUCCESS, status
    status, name = driver.cuDeviceGetName(128, device)
    assert status == driver.CUresult.CUDA_SUCCESS, status
    decoded = name.decode("utf-8") if isinstance(name, bytes) else str(name)
    decoded = decoded.rstrip("\x00 ")
    assert decoded == "NVIDIA B200", decoded
    print(f"device[{index}]={decoded}")
print(f"cuda-python={importlib.metadata.version('cuda-python')}")
print(f"cuda-bindings={importlib.metadata.version('cuda-bindings')}")
print("cuda_driver_device_smoke=PASS")
PY
