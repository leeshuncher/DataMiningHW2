#!/usr/bin/env bash
# Build the E3 upload HW2_G13.zip with the exact file names the assignment requires:
#   HW2_G13_report.pdf                 (paper/HW2_G13_report.pdf)
#   HW2_G13_presentation.pptx or .pdf  (slides/HW2_G13_presentation.*)
set -euo pipefail
cd "$(dirname "$0")/.."

report=paper/HW2_G13_report.pdf
slides=$(ls slides/HW2_G13_presentation.pptx slides/HW2_G13_presentation.pdf 2>/dev/null | head -n 1 || true)

[[ -f "$report" ]] || { echo "missing $report (build the paper first)" >&2; exit 1; }
[[ -n "$slides" ]] || { echo "missing slides/HW2_G13_presentation.pptx or .pdf" >&2; exit 1; }

rm -f HW2_G13.zip
zip -j HW2_G13.zip "$report" "$slides"
unzip -l HW2_G13.zip
