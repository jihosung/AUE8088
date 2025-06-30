import glob
import os
import subprocess
# for debug:
"""
python train_simple.py \
    --img 640 \
    --batch-size 16 \
    --epochs 20 \
    --data data/kaist-rgbt-split-byOPT.yaml \
    --cfg models/for_aug/yolov5n_kaist-rgbt-aug.yaml \
    --weights yolov5n.pt \
    --workers 8 \
    --name Yolov5n-rgbt-mosaic0.5-mixup0.2 \
    --entity $WANDB_ENTITY \
    --rgbt \
    --single-cls \
    --hyp data/hyps/for_aug/hyp.scratch-aug-mosaic-0.5-mixup-0.2.yaml \
    --optimizer SGD # SGD, Adam, AdamW\
    # --cos-lr \
    # --multi-scale
"""


# Todo: 아래 코드 aug1, aug2... 등의 hyp 파일에 대해서 적용되도록 수정하기
# hyp 파일들이 있는 디렉토리 & 패턴
hyp_dir = 'data/hyps/for_aug'
pattern = 'hyp.scratch-aug-mosaic-*.yaml'

# YAML 파일 리스트 (정렬해서 일관된 순서로 실행)
hyp_paths = sorted(glob.glob(os.path.join(hyp_dir, pattern)))
print("hyp.yaml files:", hyp_paths)

for hyp_path in hyp_paths:
    # 파일명에서 “0.2-0.7” 같은 파라미터 부분만 추출
    hyp_name = os.path.basename(hyp_path).replace('hyp.scratch-aug-mosaic-', '').replace('.yaml', '')
    run_name = f'yolov5n-rgbt-mosaic.{hyp_name}'

    # shell 커맨드 문자열 생성
    cmd = f"""
    python train_simple.py \
        --img 640 \
        --batch-size 16 \
        --epochs 20 \
        --data data/kaist-rgbt-split-byOPT.yaml \
        --cfg models/for_aug/yolov5n_kaist-rgbt-aug.yaml \
        --weights yolov5n.pt \
        --workers 8 \
        --name {run_name} \
        --entity $WANDB_ENTITY \
        --rgbt \
        --single-cls \
        --hyp {hyp_path} \
        --optimizer SGD # SGD, Adam, AdamW\
        # --cos-lr \
        # --multi-scale
    """

    print(f"🚀 Running experiment: {run_name}")
    subprocess.run(cmd, shell=True, check=True)
