import subprocess
# for debug:
"""
python train_simple.py \
    --img 640 \
    --batch-size 16 \
    --epochs 20 \
    --data data/kaist-rgbt-split-byMAN.yaml \
    --cfg models/yolov5n_kaist-rgbt.yaml \
    --weights yolov5n.pt \
    --workers 4 \
    --name debug \
    --entity $WANDB_ENTITY \
    --rgbt \
    --single-cls \
    --hyp data/hyps/hyp.scratch-low.yaml \
    --optimizer SGD # SGD, Adam, AdamW\
    # --cos-lr \
    # --multi-scale
"""

# test various dataset split

# MAN
cmd1 = """
python train_simple.py \
    --img 640 \
    --batch-size 16 \
    --epochs 20 \
    --data data/kaist-rgbt-split-byMAN.yaml \
    --cfg models/yolov5n_kaist-rgbt.yaml \
    --weights yolov5n.pt \
    --workers 4 \
    --name yolov5n-rgbt-datasetMAN \
    --entity $WANDB_ENTITY \
    --rgbt \
    --single-cls \
    --hyp data/hyps/hyp.scratch-low.yaml \
    --optimizer SGD # SGD, Adam, AdamW\
    # --cos-lr \
    # --multi-scale
"""
# OPT
cmd2 = """
python train_simple.py \
    --img 640 \
    --batch-size 16 \
    --epochs 20 \
    --data data/kaist-rgbt-split-byOPT.yaml \
    --cfg models/yolov5n_kaist-rgbt.yaml \
    --weights yolov5n.pt \
    --workers 4 \
    --name yolov5n-rgbt-datasetOPT \
    --entity $WANDB_ENTITY \
    --rgbt \
    --single-cls \
    --hyp data/hyps/hyp.scratch-low.yaml \
    --optimizer SGD # SGD, Adam, AdamW\
    # --cos-lr \
    # --multi-scale
"""
# GPT
cmd3 = """
python train_simple.py \
    --img 640 \
    --batch-size 16 \
    --epochs 20 \
    --data data/kaist-rgbt-split-byGPT.yaml \
    --cfg models/yolov5n_kaist-rgbt.yaml \
    --weights yolov5n.pt \
    --workers 4 \
    --name yolov5n-rgbt-datasetGPT \
    --entity $WANDB_ENTITY \
    --rgbt \
    --single-cls \
    --hyp data/hyps/hyp.scratch-low.yaml \
    --optimizer SGD # SGD, Adam, AdamW\
    # --cos-lr \
    # --multi-scale
"""
# 실행
print("🚀 Running first command...")
subprocess.run(cmd1, shell=True, check=True)

print("🚀 Running second command...")
subprocess.run(cmd2, shell=True, check=True)

print("🚀 Running third command...")
subprocess.run(cmd3, shell=True, check=True)
