import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.loss import v8SegmentationLoss


# 1. 定義 Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: Sigmoid 後的預測機率 [N, H, W]
        # target: 二值真實標籤 [N, H, W] (0 或 1)
        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = (pred * target).sum(dim=1)
        union = (pred.pow(2)).sum(dim=1) + (target.pow(2)).sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()


# 2. 定義結合 BCE 與 Dice 的自定義 Mask Loss
class CustomSegmentationLoss(v8SegmentationLoss):
    def __init__(self, model):
        super().__init__(model)
        self.dice_loss = DiceLoss(smooth=1.0)
        self.dice_weight = 1.0  # Dice Loss 權重，可視情況設為 0.5 ~ 1.5

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        # 原生 Prototype 解碼
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 160, 160)
        loss = 0.0

        for i, (p, g, a) in enumerate(zip(pred_mask, gt_mask, area)):
            if a > 0:
                # 原始 BCE Loss
                bce = self.bce(p, g)
                
                # 計算 Dice Loss (將 logit 經由 sigmoid 轉成機率)
                p_prob = torch.sigmoid(p)
                dice = self.dice_loss(p_prob.unsqueeze(0), g.unsqueeze(0))
                
                # 融合 BCE + Dice Loss
                loss += (bce + self.dice_weight * dice) / a
                
        return loss


# 3. 封裝自定義 Trainer
class CustomSegmentationTrainer(SegmentationTrainer):
    def get_validator(self):
        return P2SegmentationValidator(
            getattr(self, "test_loader", None),
            save_dir=self.save_dir,
            args=self.args,
            _callbacks=self.callbacks,
        )

    def get_loss(self):
        self.model.args = self.args
        return CustomSegmentationLoss(self.model)


class P2SegmentationValidator(SegmentationValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self.process = ops.process_mask_native

    def postprocess(self, preds):
        proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        predictions = DetectionValidator.postprocess(self, preds[0])
        imgsz = [self.args.mask_ratio * x for x in proto.shape[2:]]

        for index, prediction in enumerate(predictions):
            coefficients = prediction.pop("extra")
            prediction["masks"] = self.process(
                proto[index],
                coefficients,
                prediction["bboxes"],
                shape=imgsz,
            )

        return predictions