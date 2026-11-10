cd /Users/abzy/.t3/projects/small-dataset-ml
for args in "S1_frcnn S1_studio2desk 30 0" "S2_frcnn S2_mixed 30 0" "S1_frcnn_cp S1_studio2desk 10 1"; do
  uv run python src/train_frcnn.py ${=args} 2>&1 | grep -E 'all mAP|Error'
done
