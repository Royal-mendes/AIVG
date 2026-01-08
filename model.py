import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections.abc import Mapping
from typing import Optional
from torchvision.models import ResNet50_Weights
from transformers import BertModel, BertTokenizer


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert normalized xywh -> xyxy."""
    x_c, y_c, w, h = boxes.unbind(-1)
    x1 = x_c - w / 2.0
    y1 = y_c - h / 2.0
    x2 = x_c + w / 2.0
    y2 = y_c + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _clamp_boxes(boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Clamp normalized boxes to [0,1] range and enforce valid ordering."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    x1 = x1.clamp(0.0, 1.0)
    y1 = y1.clamp(0.0, 1.0)
    x2 = x2.clamp(0.0, 1.0)
    y2 = y2.clamp(0.0, 1.0)

    x_low = torch.min(x1, x2)
    x_high = torch.max(x1, x2)
    y_low = torch.min(y1, y2)
    y_high = torch.max(y1, y2)

    # prevent degenerated boxes
    w = (x_high - x_low).clamp(min=eps)
    h = (y_high - y_low).clamp(min=eps)
    center_x = (x_high + x_low) * 0.5
    center_y = (y_high + y_low) * 0.5

    x1 = (center_x - w * 0.5).clamp(0.0, 1.0)
    x2 = (center_x + w * 0.5).clamp(0.0, 1.0)
    y1 = (center_y - h * 0.5).clamp(0.0, 1.0)
    y2 = (center_y + h * 0.5).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _box_to_mask(boxes_xyxy: torch.Tensor, H: int, W: int, margin: torch.Tensor) -> torch.Tensor:
    """
    Create per-sample binary masks (B,1,H,W) from normalized xyxy boxes with adaptive margin.
    `margin` is tensor of shape [B] containing fractional expansion for each box.
    """
    B = boxes_xyxy.shape[0]
    device = boxes_xyxy.device
    ys = torch.linspace(0.0, 1.0, steps=H, device=device).view(1, H, 1).expand(1, H, W)
    xs = torch.linspace(0.0, 1.0, steps=W, device=device).view(1, 1, W).expand(1, H, W)

    masks = []
    for i in range(B):
        x1, y1, x2, y2 = boxes_xyxy[i]
        m = float(margin[i].item())
        mx1 = torch.clamp(x1 - m, 0.0, 1.0)
        my1 = torch.clamp(y1 - m, 0.0, 1.0)
        mx2 = torch.clamp(x2 + m, 0.0, 1.0)
        my2 = torch.clamp(y2 + m, 0.0, 1.0)
        mask = (xs >= mx1) & (xs <= mx2) & (ys >= my1) & (ys <= my2)
        masks.append(mask.float().squeeze(0))
    return torch.stack(masks, dim=0).unsqueeze(1)  # [B,1,H,W]


# ----------------------------------------------------------------------
# Visual backbone (ResNet50 truncated)
# ----------------------------------------------------------------------
class VisualBackbone(nn.Module):
    """ResNet50 backbone producing a 256x32x32 feature map."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.visual_feat_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.feat_downsample = nn.AdaptiveAvgPool2d((32, 32))
        self.feat_proj = nn.Conv2d(2048, 256, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.visual_feat_extractor(x)
        x = self.feat_downsample(x)
        x = self.feat_proj(x)
        return x


# ----------------------------------------------------------------------
# Language encoder (BERT)
# ----------------------------------------------------------------------
class LanguageEncoder(nn.Module):
    """
    Encode Chinese instructions using BERT base.
    Returns:
      cls_feat  : [B,768]
      seq_feat  : [B,L,768]
      lang_proj : [B,256]
      spatial_logits : [B,3]
      object_logits  : [B,3]
    """

    def __init__(self, bert_path: str = "bert-base-chinese", device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)

        # unfreeze top 4 layers for adaptation
        for p in self.bert.parameters():
            p.requires_grad = False
        for layer in self.bert.encoder.layer[-4:]:
            for p in layer.parameters():
                p.requires_grad = True

        hidden = self.bert.config.hidden_size
        self.proj_256 = nn.Linear(hidden, 256)
        self.spatial_head = nn.Linear(hidden, 3)
        self.object_head = nn.Linear(hidden, 3)

    @torch.no_grad()
    def _move_to_device(self, batch):
        for k, v in batch.items():
            batch[k] = v.to(self.device, non_blocking=True)
        return batch

    def _prepare_tokens(self, inputs, attention_mask=None, token_type_ids=None):
        if isinstance(inputs, Mapping):
            tokens = {
                k: v for k, v in inputs.items()
                if k in {"input_ids", "attention_mask", "token_type_ids"} and v is not None
            }
        elif torch.is_tensor(inputs):
            tokens = {"input_ids": inputs}
            if attention_mask is not None:
                tokens["attention_mask"] = attention_mask
            if token_type_ids is not None:
                tokens["token_type_ids"] = token_type_ids
        else:
            if isinstance(inputs, str):
                text_list = [inputs]
            else:
                text_list = list(inputs)
            tokens = self.tokenizer(
                text_list,
                padding=True,
                truncation=True,
                max_length=48,
                return_tensors="pt"
            )
        return self._move_to_device(tokens)

    def forward(self, inputs, attention_mask=None, token_type_ids=None):
        tokens = self._prepare_tokens(inputs, attention_mask, token_type_ids)
        outputs = self.bert(**tokens)
        seq_feat = outputs.last_hidden_state          # [B,L,768]
        cls_feat = seq_feat[:, 0, :]                  # [B,768]
        lang_proj = self.proj_256(cls_feat)           # [B,256]
        spatial_logits = self.spatial_head(cls_feat)  # [B,3]
        object_logits = self.object_head(cls_feat)    # [B,3]
        return cls_feat, seq_feat, lang_proj, spatial_logits, object_logits


# ----------------------------------------------------------------------
# FiLM modulation
# ----------------------------------------------------------------------
class FiLM(nn.Module):
    """Channel-wise FiLM modulation using language vector."""

    def __init__(self, channels: int = 256):
        super().__init__()
        self.gamma_mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels)
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels)
        )

    def forward(self, visual: torch.Tensor, lang_vec: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_mlp(lang_vec).view(lang_vec.size(0), -1, 1, 1)
        beta = self.beta_mlp(lang_vec).view(lang_vec.size(0), -1, 1, 1)
        return visual * (1.0 + torch.tanh(gamma)) + beta


# ----------------------------------------------------------------------
# Cross-attention (visual <-- language tokens)
# ----------------------------------------------------------------------
class CrossAttention(nn.Module):
    def __init__(self, d_model_v: int = 256, d_model_l: int = 768, nhead: int = 8):
        super().__init__()
        self.lang_proj = nn.Linear(d_model_l, d_model_v)
        self.attn = nn.MultiheadAttention(embed_dim=d_model_v, num_heads=nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model_v, d_model_v * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model_v * 2, d_model_v)
        )
        self.norm1 = nn.LayerNorm(d_model_v)
        self.norm2 = nn.LayerNorm(d_model_v)

    def forward(self, visual_tokens: torch.Tensor, lang_seq: torch.Tensor) -> torch.Tensor:
        lang_kv = self.lang_proj(lang_seq)
        attn_out, _ = self.attn(visual_tokens, lang_kv, lang_kv)
        x = self.norm1(visual_tokens + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


# ----------------------------------------------------------------------
# Pointer guidance (指示物引导)
# ----------------------------------------------------------------------
class PointerGuidance(nn.Module):
    """
    依据指示物预测框对视觉特征及语言向量做引导式调制。
    """

    def __init__(self, channels: int = 256, margin_scale: float = 0.2):
        super().__init__()
        self.margin_scale = margin_scale
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(channels + 256, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )
        self.lang_fuse = nn.Sequential(
            nn.Linear(channels + 256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )
        self.spatial_alpha = 0.6
        self.channel_alpha = 0.5
        self.lang_alpha = 0.4

    def forward(
        self,
        feat: torch.Tensor,
        lang_vec: torch.Tensor,
        pointer_xywh: torch.Tensor
    ):
        B, C, H, W = feat.shape
        boxes_xyxy = _clamp_boxes(_xywh_to_xyxy(pointer_xywh))

        w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=1e-4)
        h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=1e-4)
        diag = torch.sqrt(w * w + h * h)
        margin = torch.clamp(diag * self.margin_scale, min=0.02, max=0.2)

        mask = _box_to_mask(boxes_xyxy, H, W, margin).to(feat.dtype)
        refined_mask = self.spatial_gate(mask)  # [B,1,H,W]

        mask_area = refined_mask.flatten(2).sum(dim=2).clamp(min=1.0)
        pointer_feat = (feat * refined_mask).flatten(2).sum(dim=2) / mask_area  # [B,C]

        fusion_input = torch.cat([pointer_feat, lang_vec], dim=1)
        ch_gate = self.channel_gate(fusion_input).view(B, C, 1, 1)
        lang_delta = torch.tanh(self.lang_fuse(fusion_input))

        spatial_boost = 1.0 + self.spatial_alpha * refined_mask
        channel_boost = 1.0 + self.channel_alpha * ch_gate
        guided_feat = feat * spatial_boost * channel_boost
        guided_lang = lang_vec + self.lang_alpha * lang_delta

        return guided_feat, guided_lang, boxes_xyxy, refined_mask, pointer_feat


# ----------------------------------------------------------------------
# Simple bounding box head (predict normalized xywh)
# ----------------------------------------------------------------------
class SimpleBBoxPredictor(nn.Module):
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, 4, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.act = nn.Sigmoid()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.conv(feat)
        x = self.head(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.act(x)


# ----------------------------------------------------------------------
# Dynamic bounding box predictor with conditional modulation
# ----------------------------------------------------------------------
class DynamicBBoxPredictor(nn.Module):
    """
    Predict instruction-aware bounding boxes by combining local visual evidence
    and global conditioning (language, pointer context, relation logits).
    """

    def __init__(
        self,
        in_channels: int = 256,
        lang_dim: int = 256,
        pointer_dim: int = 256,
        meta_dim: int = 6,
        hidden: int = 256,
        pool_size: int = 4
    ):
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, 160, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 160, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.local_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.local_fc = nn.Sequential(
            nn.Linear(160 * pool_size * pool_size, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4)
        )

        cond_dim = pointer_dim + lang_dim + meta_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4),
            nn.Tanh()
        )
        self.delta_scale = 0.75

    def forward(
        self,
        feat: torch.Tensor,
        lang_vec: torch.Tensor,
        pointer_context: torch.Tensor,
        spatial_logits: Optional[torch.Tensor] = None,
        object_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B = feat.size(0)
        x = self.local_conv(feat)
        local_feat = self.local_pool(x).flatten(1)
        base_logits = self.local_fc(local_feat)

        if spatial_logits is not None:
            spatial_cond = torch.softmax(spatial_logits, dim=-1)
        else:
            spatial_cond = torch.zeros(B, 3, device=feat.device, dtype=feat.dtype)
        if object_logits is not None:
            object_cond = torch.softmax(object_logits, dim=-1)
        else:
            object_cond = torch.zeros(B, 3, device=feat.device, dtype=feat.dtype)

        cond_vec = torch.cat([pointer_context, lang_vec, spatial_cond, object_cond], dim=1)
        delta_logits = self.cond_mlp(cond_vec) * self.delta_scale

        logits = base_logits + delta_logits
        pred = torch.sigmoid(logits)
        pred = torch.clamp(pred, min=1e-4, max=1.0)
        pred_wh = torch.clamp(pred[:, 2:], min=1e-3)
        pred = torch.cat([pred[:, :2], pred_wh], dim=1)
        return pred


# ----------------------------------------------------------------------
# Bounding-box scoring head
# ----------------------------------------------------------------------
class BBoxScoreHead(nn.Module):
    """
    Estimate how well the predicted box matches instruction-conditioned features.
    Score \in [0,1].
    """

    def __init__(self, feat_channels: int = 256, lang_dim: int = 256, hidden: int = 256):
        super().__init__()
        in_features = feat_channels * 2 + lang_dim
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, feat: torch.Tensor, lang_vec: torch.Tensor, boxes_xywh: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        boxes_xyxy = _clamp_boxes(_xywh_to_xyxy(boxes_xywh))

        w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=1e-4)
        h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=1e-4)
        diag = torch.sqrt(w * w + h * h)
        margin = torch.clamp(diag * 0.25, min=0.02, max=0.18)

        mask = _box_to_mask(boxes_xyxy, H, W, margin).to(feat.dtype)
        masked_feat = (feat * mask).sum(dim=(2, 3))
        mask_area = mask.flatten(1).sum(dim=1, keepdim=True).clamp(min=1.0)

        pooled = (masked_feat / mask_area).view(B, -1)  # ensure 2-D
        global_avg = feat.flatten(2).mean(dim=2).view(B, -1)  # [B,C]

        combined = torch.cat([pooled, global_avg, lang_vec.view(B, -1)], dim=1)
        x = F.relu(self.fc1(combined), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = torch.sigmoid(self.fc3(x))
        return x


# ----------------------------------------------------------------------
# Feature refinement block
# ----------------------------------------------------------------------
class FeatureRefinement(nn.Module):
    """
    Enhance the fused feature map around current prediction guided by language vector.
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )
        self.alpha = 0.7
        self.beta = 0.5
        self.gamma = 0.2

    def forward(self, feat: torch.Tensor, lang_vec: torch.Tensor, boxes_xywh: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        boxes_xyxy = _clamp_boxes(_xywh_to_xyxy(boxes_xywh))

        w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=1e-4)
        h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=1e-4)
        diag = torch.sqrt(w * w + h * h)
        margin = torch.clamp(diag * 0.3, min=0.03, max=0.2)

        mask = _box_to_mask(boxes_xyxy, H, W, margin).to(feat.dtype)
        refined_mask = self.spatial_refine(mask)

        ch_gate = self.channel_gate(lang_vec).view(B, C, 1, 1)
        spatial_boost = 1.0 + self.alpha * refined_mask
        channel_boost = 1.0 + self.beta * ch_gate
        blended = feat * spatial_boost * channel_boost
        return feat * (1.0 - self.gamma) + blended * self.gamma


# ----------------------------------------------------------------------
# Main model with adaptive iterative enhancement
# ----------------------------------------------------------------------
class PDFWordLocator(nn.Module):
    """
    Visual grounding model with adaptive iterative refinement.
    - Initial fusion via FiLM + cross-attention.
    - Iteratively predicts a bounding box, scores it, and refines feature map.
    - Loop stops after max_iters, or early if score fails to improve twice consecutively (per batch element).
    """

    def __init__(
        self,
        bert_path: str = "bert-base-chinese",
        device: torch.device = torch.device("cpu"),
        min_iters: int = 2,
        max_iters: int = 5
    ):
        super().__init__()
        assert max_iters >= min_iters >= 1
        self.device = device
        self.min_iters = int(min_iters)
        self.max_iters = int(max_iters)

        self.visual_backbone = VisualBackbone()
        self.lang_encoder = LanguageEncoder(bert_path=bert_path, device=device)
        self.pointer_head = SimpleBBoxPredictor(in_channels=256)
        self.pointer_guidance = PointerGuidance(channels=256, margin_scale=0.2)
        self.film = FiLM(channels=256)
        self.cross_attn = CrossAttention(d_model_v=256, d_model_l=768, nhead=8)
        self.bbox_head = DynamicBBoxPredictor(
            in_channels=256,
            lang_dim=256,
            pointer_dim=256,
            meta_dim=6,
            hidden=256,
            pool_size=4
        )
        self.score_head = BBoxScoreHead(feat_channels=256, lang_dim=256, hidden=256)
        self.refiner = FeatureRefinement(channels=256)

    def forward(
        self,
        images: torch.Tensor,
        texts,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ):
        B = images.size(0)

        visual_feat = self.visual_backbone(images)  # [B,256,32,32]
        pointer_bbox = self.pointer_head(visual_feat)  # [B,4]

        cls_feat, seq_feat, lang_vec_raw, spatial_logits, object_logits = self.lang_encoder(
            texts,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        spatial_probs = torch.softmax(spatial_logits, dim=-1)
        object_probs = torch.softmax(object_logits, dim=-1)

        guided_feat, lang_vec_guided, pointer_bbox_xyxy, pointer_mask, pointer_context = self.pointer_guidance(
            visual_feat,
            lang_vec_raw,
            pointer_bbox
        )

        film_feat = self.film(guided_feat, lang_vec_guided)
        tokens = film_feat.flatten(2).transpose(1, 2)
        fused_tokens = self.cross_attn(tokens, seq_feat)
        fused_feat = fused_tokens.transpose(1, 2).reshape(B, 256, 32, 32)

        iter_boxes = []
        iter_boxes_xyxy = []
        iter_scores = []
        history = []
        best_score = None
        best_bbox = None
        best_bbox_xyxy = None
        patience = torch.zeros(B, dtype=torch.long, device=images.device)
        score_eps = 1e-4

        feat_iter = fused_feat
        for it in range(self.max_iters):
            boxes_xywh = self.bbox_head(
                feat_iter,
                lang_vec_guided,
                pointer_context,
                spatial_logits=spatial_logits,
                object_logits=object_logits
            )
            boxes_xyxy = _clamp_boxes(_xywh_to_xyxy(boxes_xywh))
            scores = self.score_head(feat_iter, lang_vec_guided, boxes_xywh)

            iter_boxes.append(boxes_xywh)
            iter_boxes_xyxy.append(boxes_xyxy)
            iter_scores.append(scores)

            history.append({
                "iter": it + 1,
                "bbox_xywh": boxes_xywh.detach(),
                "bbox_xyxy": boxes_xyxy.detach(),
                "score": scores.detach()
            })

            if best_score is None:
                best_score = scores
                best_bbox = boxes_xywh
                best_bbox_xyxy = boxes_xyxy
                improved_mask = torch.ones(B, dtype=torch.bool, device=images.device)
            else:
                improved_mask = scores.squeeze(1) > (best_score.squeeze(1) + score_eps)
                mask = improved_mask.unsqueeze(1)
                best_score = torch.where(mask, scores, best_score)
                best_bbox = torch.where(mask.expand_as(best_bbox), boxes_xywh, best_bbox)
                best_bbox_xyxy = torch.where(mask.expand_as(best_bbox_xyxy), boxes_xyxy, best_bbox_xyxy)

            patience = torch.where(improved_mask, torch.zeros_like(patience), patience + 1)

            if it + 1 >= self.min_iters and torch.all(patience >= 2):
                break

            feat_iter = self.refiner(feat_iter, lang_vec_guided, boxes_xywh)

        iter_boxes = torch.stack(iter_boxes, dim=0)           # [T,B,4]
        iter_boxes_xyxy = torch.stack(iter_boxes_xyxy, dim=0) # [T,B,4]
        iter_scores = torch.stack(iter_scores, dim=0)         # [T,B,1]

        outputs = {
            "final_bbox": best_bbox,                 # xywh in [0,1]
            "final_bbox_xyxy": best_bbox_xyxy,       # xyxy in [0,1]
            "best_score": best_score,                # [B,1]
            "iter_boxes": iter_boxes,                # [T,B,4] (xywh)
            "iter_boxes_xyxy": iter_boxes_xyxy,      # [T,B,4] (xyxy)
            "iter_scores": iter_scores,              # [T,B,1]
            "iterations_used": iter_boxes.size(0),
            "history": history,
            "visual_feat": visual_feat.detach(),
            "guided_visual_feat": guided_feat.detach(),
            "fused_feat": fused_feat.detach(),
            "pointer_bbox": pointer_bbox,
            "pointer_bbox_xyxy": pointer_bbox_xyxy,
            "pointer_mask": pointer_mask.detach(),
            "pointer_context": pointer_context.detach(),
            "lang_vec_raw": lang_vec_raw.detach(),
            "lang_vec_guided": lang_vec_guided.detach(),
            "lang_vec": lang_vec_guided.detach(),
            "spatial_logits": spatial_logits,
            "object_logits": object_logits,
            "spatial_probs": spatial_probs,
            "object_probs": object_probs,
            "spatial_pred": spatial_probs.argmax(dim=-1).detach(),
            "object_pred": object_probs.argmax(dim=-1).detach()
        }
        return outputs


# Backward-compatible alias
AdaptivePDFLocator = PDFWordLocator
