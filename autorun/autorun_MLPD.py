import subprocess

# 첫 번째 명령어 (문자 그대로)
cmd1 = """
python train_simple.py \
    --img 640 \
    --batch-size 16 \
    --epochs 200 \
    --data data/kaist-rgbt-split-byOPT.yaml \
    --cfg models/yolov5n_kaist-rgbt-fromMLPD1.yaml \
    --weights '' \
    --workers 8 \
    --name Yolov5n-rgbt-MLPD1 \
    --entity $WANDB_ENTITY \
    --rgbt \
    --single-cls \
    --hyp data/hyps/for_MLPD/hyp.scratch-MLPD.yaml \
    --optimizer SGD # SGD, Adam, AdamW\
    # --cos-lr \
    # --multi-scale
"""

# 두 번째 명령어
cmd2 = """
python train_simple.py \
    --img 640 \
    --batch-size 16 \
    --epochs 200 \
    --data data/kaist-rgbt-split-byOPT.yaml \
    --cfg models/yolov5n_kaist-rgbt-fromMLPD2.yaml \
    --weights '' \
    --workers 8 \
    --name Yolov5n-rgbt-MLPD2 \
    --entity $WANDB_ENTITY \
    --rgbt \
    --single-cls \
    --hyp data/hyps/for_MLPD/hyp.scratch-MLPD.yaml \
    --optimizer SGD # SGD, Adam, AdamW\
    # --cos-lr \
    # --multi-scale
"""

# 실행
print("🚀 Running first command...")
subprocess.run(cmd1, shell=True, check=True)

# print("🚀 Running second command...")
# subprocess.run(cmd2, shell=True, check=True)
