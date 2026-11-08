#!/bin/bash
# Dataset cleanup: only what training needs stays (4K mania difficulties + their audio; no video, bg, hitsounds, clones)
python3 audit_dataset_compatibility.py --clean --skip_audit "$@"
read -r -p "Remove all of it now? (y/N): " ok
if [ "$ok" = "y" ] || [ "$ok" = "Y" ]; then
  python3 audit_dataset_compatibility.py --clean --apply "$@"
fi
