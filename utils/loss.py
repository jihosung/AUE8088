# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Loss functions."""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.metrics import bbox_iop
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441"""
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

# 실제 로스 계산하는 클래스
class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    # 실제 loss 계산하는 함수 p(prediction), target(GT)를 전달받음
    def __call__(self, p, targets):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss

        """
        모델 구조
        P3 feature map이 있다고 하자. P3에서 나오는 출력값은 pi(P3에 해당하는 index i)이다.
        출력값 pi는 regression(sigmoid + scaling)을 거쳐서 다음 layer로 간다.
        여기서 loss 계산을 해주는데, iou를 사용한다.
        iou 계산에 쓰일 값은 "regression을 거친 값"과 P3의 scale로 변환된 GT이다.
        학습이 지속될수록 P3 내부 파라미터들이 업데이트되고, pi가 regression된 값은 scaled GT에 수렴할것이다.
            --------------------> P3 layer --------------------------------> regression --------------------------------------------> next layer
            이전 layer 출력                        pi:GT와 간접적인 관계                      예측값: GT와 직접적으로 비교 가능한 예측값

        build_targets의 역할
        1. p: 약 300개의 dense한 예측. YOLO v5 모델은 예측을 많이 생성해냄
        2. 이중에서 GT랑 가까운 예측을 선별, 예측과 GT를 1:1 매칭해줌: 1:1 매칭 과정에서 1개의 GT에 여러개의 예측이 몰린다면 GT를 복사해서 개수 맞춰줌
            2-1. 그래서 iou 계산에서 1번 tbox랑 1번 pbox만 계산하면 되는것
        3. 매칭된 GT와 예측을 가지고 loss 계산
        """
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # target class, target box, prediction indices, prediction anchor size

        # Losses 계산 루프, 루프 단위: 
        for i, pi in enumerate(p):  # layer index, layer predictions
            # i: i번째 feature map
            # pi: i번째 feature map에서의 예측값

            # 인덱스
            """
            인덱스, 왜 필요한가?
            YOLO 모델은 한번에 많은 예측값을 만듦(ex. 300개) -> 예측의 밀도가 높다
            이 예측값들 중 GT에 근접한 아이들만 선별 & 1대1 매칭: build_targets 함수의 역할
            GT와 매칭된 예측값들을 구분하기 위해 아래 index를 씀
            batch 16개 중 b번째 이미지에, 예측에 사용한 anchor 번호는 a이고, 예측값의 좌표는 gj, gi이다

            anchor 번호 예시:
                - 현재 i번째 layer의 앵커 세트: [[10,13], [16,30], [33,23]]  → index 0,1,2
                - anchor: tensor([0, 0, 1, 1, 1]) 0, 0, 1, 1, 1번째 anchor가 예측에 사용되었다는 뜻
            """
            b, a, gj, gi = indices[i]  # 이번 i번째 layer에서 예측값들 index 받아오기

            # target obj: object 검출 시 맞을 확률 목표 (iou 기준)
            # 초기값: 0
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  

            n = b.shape[0]  # number of targets
            if n: # b가 있다면 = i_th layer의 예측값이 존재한다면

                # 1. 예측값 pi에서 해당 prediction 인덱싱 후 split으로 필요한 정보만 받아옴
                pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions
                

                # 2. box loss
                # 2-1. Regression
                """
                regression이 주는 의미???
                과정에서 이해해보기
                ┌──────────────────────────────[ Stage 1 ]──────────────────────────────┐
                │                     Feature Map에서 나온 원시 예측값 (Raw Logits)        │
                │                                                                       │
                │ 예: pi[b, a, gj, gi] = [ raw_x, raw_y, raw_w, raw_h, obj, cls... ]    │
                │      ↑ 앵커 기준 상대 좌표, 제한 없는 실수 값                             │
                │      ↑ 아직은 GT(label) 위치와 직접적인 비교 불가능                       │
                └──────────────────────────────────────────────────────────────────────┘

                ┌──────────────────────────────[ Stage 2 ]──────────────────────────────┐
                │                          회귀 수식 (Regression) 적용                    │
                │                                                                       │
                │ pxy = sigmoid(raw_xy) * 2 - 0.5       ← 그리드 셀 내부 상대 위치         │
                │ pwh = (sigmoid(raw_wh) * 2) ** 2 * anchor  ← 앵커 기반 width/height   │
                │ pbox = torch.cat((pxy, pwh), dim=1)                                  │
                │                                                                      │
                │ ✅ 이제 예측 box는 grid 셀 단위로 표현됨 (예: x=10.4, y=7.2 등)          │
                └──────────────────────────────────────────────────────────────────────┘

                ┌──────────────────────────────[ Stage 3 ]──────────────────────────────┐
                │                       GT(Label)도 동일한 좌표계로 변환 필요            │
                │                                                                        │
                │ GT 원본: YOLO 형식 [cls, x, y, w, h] (모두 0~1 정규화)               │
                │                                                                        │
                │ build_targets() 내부에서                                               │
                │     → GT * gain (grid 크기 곱하기)                                    │
                │     → GT도 grid 셀 기준 좌표로 스케일 업                              │
                │                                                                        │
                │ ✅ 이제 예측값(pbox)과 GT(tbox)가 동일한 단위로 비교 가능            │
                └──────────────────────────────────────────────────────────────────────┘

                ┌──────────────────────────────[ Stage 4 ]──────────────────────────────┐
                │                            IoU 계산 및 Loss                            │
                │                                                                        │
                │ iou = bbox_iou(pbox, tbox, CIoU=True)                                  │
                │ loss_box += (1.0 - iou).mean()                                         │
                │                                                                        │
                │ 최종적으로 학습 시에는 regression된 예측값과 변환된 GT를              │
                │ 같은 공간에서 비교하여 손실 계산                                       │
                └──────────────────────────────────────────────────────────────────────┘
                """
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # make predicted box

                # 2-2. iou 계산
                # iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # original code
                iou = bbox_iou(pbox, tbox[i], CIoU=True).view(-1)  # bug fixed. ex) iou가 [0.1]일때 0.1로 만들어버리는것 해결

                # 2-3. tcls[i]에서 ignore(-1) 아닌 것만 loss에 반영
                valid_idx = tcls[i] >= 0              # True for classes you care about
                if valid_idx.sum():                   # 하나라도 남았을 때만
                    iou_person = iou[valid_idx]
                    lbox += (1.0 - iou_person).mean()
                # valid_idx.sum()==0 이면 이 레이어의 box loss 스킵
                lbox += (1.0 - iou).mean()  # iou loss

                # 3. Object loss (Confidence loss)
                # - sort and gr: iou가 tobj에 미치는 비율 (보통 gr = 1)
                # - iop도 같이 추가
                iop = bbox_iop(pbox, tbox[i], CIoU=True).view(-1)
                iop = iop.detach().clamp(0).type(tobj.dtype)
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou, iop = b[j], a[j], gj[j], gi[j], iou[j], iop[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                    iop = (1.0 - self.gr) + self.gr * iop

                # 3-1. -1 class에 대해서 confidence score 무시하기
                # 왜? IoP? 아예 무시는 안되나?
                #   - 네. 아예 무시해버리면 사람 모양을 가졌지만 배경으로 학습시키는 꼴이 되어 실제 사람 학습에도 악영향을 끼칠 수 있음
                #   - 그래서 적당히 사람 모양이면 사람인가? 하는 판단은 주고, 너무 많이 그쪽으로 치우치면 무시하도록
                #   - people class에 대해 너무 치우치는 기준을 주기 위해(기존 iou의 한계점 발생 - 일정 이상 iou가 안올라감) IoP 도입
                """
                Todo:
                IoU & IoP 다 계산해서 둘 중 하나라도 thres 넘기면 tobj로 등록하지 말기(아예 conf = 0 수렴을 위함)
                thres 적당한값은... 모델이 정답이라 내뱉는 confidence 기준이 0.6
                학습하면 안되지만 그래도 사람 비슷하게 생겼으니 아예 무시하지는 말라는 의미에서,
                thres 넘지 않는 것들은 "그정도 확률로 사람이다"는걸 학습시키는것임
                    -> 사람 아니면서 사람모양이니까 대충 0.3 근처가 적당하지 않을까? 생각함
                """
                ign_idx = (tcls[i] == -1) & ((iou > self.hyp["iou_t"]) | (iop > self.hyp["iop_t"]))
                keep = ~ign_idx
                
                # 추가: bug detect
                try:
                    b, a, gj, gi, iou = b[keep], a[keep], gj[keep], gi[keep], iou[keep]
                except Exception as e:
                    print("🔥 Error during loss computation")
                    print("ign_idx:", ign_idx.shape)
                    print("keep:", keep)
                    print("[b,a,gj,ji,iou] = ", b.shape, " ",a.shape, " ", gj.shape, " ", gi.shape, " ", iou.shape)
                    raise e  # 또는 sys.exit(1)으로 강제 종료
                
                tobj[b, a, gj, gi] = iou  # iou ratio

                # 4. Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            
            # object loss 업데이트: 모델 예측 confidence score가 iou or iop를 따라가도록!
            # pi[...,4]: confidence값
            # tobj: target confidence값
            obji = self.BCEobj(pi[..., 4], tobj) 
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
