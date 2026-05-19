#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <YYYYMMDD> <config.json> [extra run_night_pipeline args...]" >&2
  exit 1
fi

night="$1"
base_config="$2"
shift 2

if [[ ! "$night" =~ ^[0-9]{8}$ ]]; then
  echo "Night must be an AudioMoth date token like 20260518." >&2
  exit 1
fi

if [[ ! -f "$base_config" ]]; then
  echo "Config file not found: $base_config" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
temp_config="$(mktemp "/tmp/batpipe-${night}-XXXXXX.json")"

python3 - <<'PY' "$base_config" "$temp_config" "$night"
from pathlib import Path
import json
import sys

source_path = Path(sys.argv[1]).expanduser().resolve()
target_path = Path(sys.argv[2])
night = sys.argv[3]

payload = json.loads(source_path.read_text(encoding="utf-8"))

night_runs_root_value = payload.get("night_runs_dir")
if night_runs_root_value not in (None, ""):
  night_runs_root = Path(str(night_runs_root_value)).expanduser().resolve()
else:
  work_root_value = payload.get("work_root_dir")
  if work_root_value not in (None, ""):
    night_runs_root = Path(str(work_root_value)).expanduser().resolve() / "night-runs"
  elif payload.get("detection_output_dir") not in (None, ""):
    night_runs_root = Path(str(payload["detection_output_dir"])).expanduser().resolve().parent / "night-runs"
  else:
    raise ValueError("Config must define night_runs_dir, work_root_dir, or detection_output_dir.")

work_root = night_runs_root / night

payload["night_token"] = night
payload["night_start_hour"] = int(payload.get("night_start_hour", 18))
payload["night_end_hour"] = int(payload.get("night_end_hour", 12))
payload["name_contains"] = []
payload["subset_limit"] = payload.get("subset_limit")
payload["night_runs_dir"] = str(night_runs_root)
payload["detection_output_dir"] = str(work_root / "detections")
payload["summary_output_dir"] = str(work_root)
payload["review_output_dir"] = str(work_root)
payload.pop("validation_output_dir", None)

target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(target_path)
PY

cleanup() {
  rm -f "$temp_config"
}
trap cleanup EXIT

cd "$repo_root"
PYTHONPATH=src python scripts/run_night_pipeline.py --config "$temp_config" "$@"