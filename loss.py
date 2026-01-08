import torch
import torch.nn as nn
import torch.nn.functional as F


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = boxes.unbind(-1)
    x1 = x_c - w / 2.0
    y1 = y_c - h / 2.0
    x2 = x_c + w / 2.0
    y2 = y_c + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _bbox_iou_xyxy(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xa1, ya1, xa2, ya2 = pred.unbind(-1)
    xb1, yb1, xb2, yb2 = tgt.unbind(-1)

    inter_x1 = torch.max(xa1, xb1)
    inter_y1 = torch.max(ya1, yb1)
    inter_x2 = torch.min(xa2, xb2)
    inter_y2 = torch.min(ya2, yb2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_a = (xa2 - xa1).clamp(min=0) * (ya2 - ya1).clamp(min=0)
    area_b = (xb2 - xb1).clamp(min=0) * (yb2 - yb1).clamp(min=0)
    union = (area_a + area_b - inter).clamp(min=eps)
    return inter / union


class PDFLocatorLoss(nn.Module):
    """
    Loss for relation-aware adaptive locator.
      - SmoothL1 + IoU on final best box
      - Score supervision: target IoU for each iteration
      - Progress regulariser: penalise IoU drops
      - Spatial/Object classification heads (ignore samples with label < 0)
    """

    def __init__(self, loss_weights=None, progress_weight: float = 0.2):
        super().__init__()
        self.loss_weights = loss_weights or {
            "bbox_l1": 1.0,
            "bbox_iou": 1.0,
            "score": 1.0,
            "progress": progress_weight,
            "spatial_cls": 1.0,
            "object_cls": 1.0,
            "pointer_l1": 1.0,
            "pointer_iou": 1.0,
        }
        self.loss_weights.setdefault("pointer_l1", 1.0)
        self.loss_weights.setdefault("pointer_iou", 1.0)
        self.l1 = nn.SmoothL1Loss(reduction="none")
        self.mse = nn.MSELoss(reduction="none")
        self.progress_weight = progress_weight

        self.spatial_map = {0: "this", 1: "prev", 2: "next"}
        self.object_map = {0: "word", 1: "line", 2: "paragraph"}
        self.fallback_keys = [
            "this_word_bbox", "this_line_bbox", "this_paragraph_bbox",
            "prev_word_bbox", "prev_line_bbox", "prev_paragraph_bbox",
            "next_word_bbox", "next_line_bbox", "next_paragraph_bbox",
        ]

    @staticmethod
    def _valid_box_xywh(box: torch.Tensor, tol: float = 1e-5) -> bool:
        if box is None or box.numel() != 4:
            return False
        return bool((box[2] > tol) and (box[3] > tol))

    def _select_gt(self, targets):
        spatial = targets["spatial_label"]
        obj = targets["object_label"]
        B = spatial.numel()
        device = spatial.device
        gt = torch.zeros(B, 4, device=device, dtype=torch.float32)
        valid = torch.zeros(B, dtype=torch.bool, device=device)

        for i in range(B):
            pick_key = None
            s = int(spatial[i].item())
            o = int(obj[i].item())
            if 0 <= s <= 2 and 0 <= o <= 2:
                key = f"{self.spatial_map[s]}_{self.object_map[o]}_bbox"
                if self._valid_box_xywh(targets[key][i]):
                    pick_key = key
            if pick_key is None:
                for cand in self.fallback_keys:
                    if self._valid_box_xywh(targets[cand][i]):
                        pick_key = cand
                        break

            if pick_key is not None:
                gt[i] = targets[pick_key][i]
                valid[i] = True
        return gt, valid

    def _relation_loss(self, logits, labels):
        if logits is None or labels is None:
            device = logits.device if logits is not None else (
                labels.device if isinstance(labels, torch.Tensor) else torch.device("cpu")
            )
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero

        device = logits.device
        labels = labels.to(device)
        valid_mask = labels >= 0
        if valid_mask.any():
            valid_logits = logits[valid_mask]
            valid_labels = labels[valid_mask]
            loss_val = F.cross_entropy(valid_logits, valid_labels, reduction="mean")
            pred = valid_logits.argmax(dim=-1)
            acc = (pred == valid_labels).float().mean()
            valid_ratio = valid_mask.float().mean()
            return loss_val, acc, valid_ratio

        zero = torch.tensor(0.0, device=device)
        valid_ratio = valid_mask.float().mean() if valid_mask.numel() > 0 else zero
        return zero, zero, valid_ratio

    def forward(self, outputs, targets):
        loss = {}
        pred_xywh = outputs["final_bbox"]
        gt_xywh, valid_mask = self._select_gt(targets)

        if valid_mask.any():
            idx = valid_mask.nonzero(as_tuple=False).squeeze(1)
            pred_valid = pred_xywh.index_select(0, idx)
            gt_valid = gt_xywh.index_select(0, idx)

            l1 = self.l1(pred_valid, gt_valid).mean(dim=-1)
            loss["bbox_l1"] = l1.mean()

            pred_xyxy = _xywh_to_xyxy(pred_valid)
            gt_xyxy = _xywh_to_xyxy(gt_valid)
            iou = _bbox_iou_xyxy(pred_xyxy, gt_xyxy)
            loss["bbox_iou"] = (1.0 - iou).mean()
        else:
            device = pred_xywh.device
            loss["bbox_l1"] = torch.tensor(0.0, device=device)
            loss["bbox_iou"] = torch.tensor(0.0, device=device)

        pointer_pred = outputs.get("pointer_bbox")
        pointer_pred_xyxy = outputs.get("pointer_bbox_xyxy")
        pointer_gt = targets.get("finger_bbox")

        if pointer_pred is not None and pointer_gt is not None:
            pointer_valid = (pointer_gt[:, 2] > 1e-5) & (pointer_gt[:, 3] > 1e-5)
            if pointer_valid.any():
                idx = pointer_valid.nonzero(as_tuple=False).squeeze(1)
                pred_ptr = pointer_pred.index_select(0, idx)
                gt_ptr = pointer_gt.index_select(0, idx)
                l1_ptr = self.l1(pred_ptr, gt_ptr).mean(dim=-1)
                loss["pointer_l1"] = l1_ptr.mean()

                if pointer_pred_xyxy is None:
                    ptr_pred_xyxy = _xywh_to_xyxy(pred_ptr)
                else:
                    ptr_pred_xyxy = pointer_pred_xyxy.index_select(0, idx)
                ptr_gt_xyxy = _xywh_to_xyxy(gt_ptr)
                ptr_iou = _bbox_iou_xyxy(ptr_pred_xyxy, ptr_gt_xyxy)
                loss["pointer_iou"] = (1.0 - ptr_iou).mean()
                loss["pointer_valid_ratio"] = pointer_valid.float().mean()
            else:
                device = pointer_pred.device
                loss["pointer_l1"] = torch.tensor(0.0, device=device)
                loss["pointer_iou"] = torch.tensor(0.0, device=device)
                loss["pointer_valid_ratio"] = pointer_valid.float().mean()
        else:
            device = pred_xywh.device
            loss["pointer_l1"] = torch.tensor(0.0, device=device)
            loss["pointer_iou"] = torch.tensor(0.0, device=device)
            loss["pointer_valid_ratio"] = torch.tensor(0.0, device=device)

        iter_boxes = outputs.get("iter_boxes")
        iter_scores = outputs.get("iter_scores")

        if iter_boxes is not None and iter_scores is not None and valid_mask.any():
            idx = valid_mask.nonzero(as_tuple=False).squeeze(1)
            gt_valid = gt_xywh.index_select(0, idx)
            gt_xyxy = _xywh_to_xyxy(gt_valid)

            scores = iter_scores[:, idx, :]
            boxes = iter_boxes[:, idx, :]
            boxes_xyxy = _xywh_to_xyxy(boxes)

            steps = iter_boxes.size(0)
            gt_expand = gt_xyxy.unsqueeze(0).expand(steps, -1, -1).reshape(-1, 4)
            pred_flat = boxes_xyxy.view(-1, 4)
            ious = _bbox_iou_xyxy(pred_flat, gt_expand).view(steps, -1, 1)

            score_err = self.mse(scores, ious).mean()
            loss["score"] = score_err

            if steps > 1 and self.progress_weight > 0:
                iou_diff = F.relu(ious[:-1] - ious[1:])
                loss["progress"] = iou_diff.mean()
            else:
                loss["progress"] = torch.tensor(0.0, device=pred_xywh.device)
        else:
            device = pred_xywh.device
            loss["score"] = torch.tensor(0.0, device=device)
            loss["progress"] = torch.tensor(0.0, device=device)

        spatial_logits = outputs.get("spatial_logits")
        object_logits = outputs.get("object_logits")
        spatial_labels = targets.get("spatial_label")
        object_labels = targets.get("object_label")

        spatial_loss, spatial_acc, spatial_valid = self._relation_loss(spatial_logits, spatial_labels)
        object_loss, object_acc, object_valid = self._relation_loss(object_logits, object_labels)
        loss["spatial_cls"] = spatial_loss
        loss["object_cls"] = object_loss
        loss["spatial_acc"] = spatial_acc
        loss["object_acc"] = object_acc
        loss["spatial_valid_ratio"] = spatial_valid
        loss["object_valid_ratio"] = object_valid

        total = torch.tensor(0.0, device=pred_xywh.device)
        for k, w in self.loss_weights.items():
            total = total + w * loss[k]
        loss["total_loss"] = total
        loss["valid_ratio"] = valid_mask.float().mean()
        return loss
