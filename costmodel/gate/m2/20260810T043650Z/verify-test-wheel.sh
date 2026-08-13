#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 PYTHON WHEEL VENV WORK SOURCE EXTRAS" >&2
  exit 2
fi

python_bin=$1
wheel=$2
venv=$3
work=$4
source_root=$5
extras=$6

export PIP_DISABLE_PIP_VERSION_CHECK=1

"${python_bin}" -m venv "${venv}"
"${venv}/bin/python" --version
"${venv}/bin/python" -m pip install "${wheel}[${extras}]"
"${venv}/bin/python" -m pip check
"${venv}/bin/python" -m pip freeze --all
"${venv}/bin/python" - <<'PY'
import sys

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
assert cm.COST_MODEL_API_VERSION == (2, 0)
assert cm.__file__ is not None and "site-packages" in cm.__file__, cm.__file__
print(f"root={cm.__file__}")
print(f"version={cm.__version__}")
print("new_optional_or_legacy_modules=[]")
PY

mkdir -p "${work}"
cp -R "${source_root}/tests" "${work}/tests"
cp -R "${source_root}/schemas" "${work}/schemas"
cp -R "${source_root}/calibration" "${work}/calibration"
cd "${work}"
"${venv}/bin/python" -m pytest tests --cov=tilefoundry_costmodel --cov-branch
