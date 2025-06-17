import subprocess

# 첫 번째 명령어 (문자 그대로)
cmd1 = """
python train_simple.py \
    --img 640 \
    --batch-size 16 \
    --epochs 20 \
    --data data/kaist-rgbt-split.yaml \
    --cfg models/yolov5n_kaist-rgbt.yaml \
    --weights yolov5n.pt \
    --workers 4 \
    --name yolov5n-rgbt-forTune-dataAug-lowHyp-newIoUAnchor \
    --entity $WANDB_ENTITY \
    --rgbt \
    --single-cls
"""

# 두 번째 명령어
cmd2 = """
python train_simple.py \
    --img 640 \
    --batch-size 16 \
    --epochs 20 \
    --data data/kaist-rgbt.yaml \
    --cfg models/yolov5n_kaist-rgbt.yaml \
    --weights yolov5n.pt \
    --workers 4 \
    --name yolov5n-rgbt-forSubmit-dataAug-lowHyp-newIoUAnchor \
    --entity $WANDB_ENTITY \
    --rgbt \
    --single-cls
"""

# 실행
print("🚀 Running first command...")
subprocess.run(cmd1, shell=True, check=True)

print("🚀 Running second command...")
subprocess.run(cmd2, shell=True, check=True)
