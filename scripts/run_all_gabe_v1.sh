#!/usr/bin/env bash
# Run from repo root: ./scripts/run_all_gabe_v1.sh
# If using uv: uv run ./scripts/run_all_gabe_v1.sh (or ensure python has jtap_mice)
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
ASSETS="${REPO_ROOT}/assets/stimuli/gabe_v1"
PYSCRIPT="${REPO_ROOT}/scripts/run_jtap_single_json.py"
OUT_DIR="${REPO_ROOT}/gabe_v1_plots_$(date +%Y%m%d_%H%M)"
LEFT_OCC="4.1,3.2"
RIGHT_OCC="5.7,3.2"
N=0
TOTAL=6
run_one() {
  local name="$1"
  local occ="$2"
  N=$((N + 1))
  local p="${ASSETS}/${name}.json"
  if [[ ! -f "$p" ]]; then echo "Skip: $p"; return 0; fi
  echo ""
  echo "[$N/$TOTAL] Running: $name"
  if [[ -z "$occ" ]]; then python "$PYSCRIPT" "$p" --output-dir "$OUT_DIR"
  else python "$PYSCRIPT" "$p" --output-dir "$OUT_DIR" --occlusion-regions "$occ"; fi
}
run_one "leftOcc_noSwitch" "$LEFT_OCC"
run_one "leftOcc_switch" "$LEFT_OCC"
run_one "rightOcc_noSwitch" "$RIGHT_OCC"
run_one "rightOcc_switch" "$RIGHT_OCC"
run_one "noOcc_noSwitch" ""
run_one "noOcc_switch" ""
echo ""
echo "Done. Outputs in $OUT_DIR"
