#!/bin/zsh
# Sequential detection training queue. Each line: [SEED=n] name split model epochs imgsz [extra]
# Lines whose name starts with "frcnn:" run Faster R-CNN: frcnn: name split epochs use_synth
cd "$(dirname $0)/.."
while read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  echo ">>> $line"
  seed=0
  if [[ "$line" == SEED=* ]]; then seed=${${line%% *}#SEED=}; line=${line#* }; fi
  if [[ "$line" == frcnn:* ]]; then
    SEED=$seed uv run python src/train_frcnn.py ${=${line#frcnn: }} 2>&1 | grep -E 'all mAP|Error'
  else
    SEED=$seed uv run python src/train_yolo.py ${=line} 2>&1 | grep -E '^[A-Za-z0-9_]+: all mAP|Error|error'
  fi
done < "$1"
