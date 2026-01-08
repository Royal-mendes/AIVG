import os
import json
import re
import random
from pathlib import Path
from collections import Counter

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer

from model import PDFWordLocator
from loss import PDFLocatorLoss


# ----------------------------
# 路径与数据采样配置（仅数据加载与1103Vs不同）
# ----------------------------
PDF_DATASET_DIR = Path("/data1/shx/loc_vg/pdf_dataset")
PDF_TRAIN_LIMIT = 16000
PDF_TEST_LIMIT = 4000

PEN_DATASET_DIR = Path("/data1/shx/loc_vg/pdf_dataset_pen")

DATA_SAMPLE_SEED = 1209
DATA_SOURCES = [
    {
        "name": "pdf_dataset",
        "annotation_dir": PDF_DATASET_DIR / "pawl_coord_dataset",
        "image_dir": PDF_DATASET_DIR / "pawl_overlay_dataset",
        "train_limit": PDF_TRAIN_LIMIT,
        "test_limit": PDF_TEST_LIMIT,
    },
    {
        "name": "pdf_dataset_pen",
        "annotation_dir": PEN_DATASET_DIR / "pawl_coord_dataset",
        "image_dir": PEN_DATASET_DIR / "pawl_overlay_dataset",
        "train_limit": None,
        "test_limit": None,
    },
]

SAVE_DIR = Path("./saved_models")


# ----------------------------
# 工具函数
# ----------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _xyxy_to_xywh(b):
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = [float(x) for x in b]
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    xc = x0 + w / 2.0
    yc = y0 + h / 2.0
    return [
        max(0.0, min(1.0, xc)),
        max(0.0, min(1.0, yc)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    ]


_SPATIAL_PATTERNS = [
    (0, r"(这个|本|此|该)"),
    (1, r"(上个|上一个|上一|上文|上一段|上一行)"),
    (2, r"(下个|下一个|下一|下文|下一段|下一行)"),
]
_OBJECT_PATTERNS = [
    (0, r"(单词|词)"),
    (1, r"(行|这一行|本行)"),
    (2, r"(段落|这一段|本段|段)"),
]


def parse_labels_from_text(text: str):
    if not isinstance(text, str):
        return -1, -1
    text = text.strip().replace(" ", "")
    spatial = -1
    for val, pat in _SPATIAL_PATTERNS:
        if re.search(pat, text):
            spatial = val
            break
    obj = -1
    for val, pat in _OBJECT_PATTERNS:
        if re.search(pat, text):
            obj = val
            break
    return spatial, obj


# ----------------------------
# 数据采样与数据集
# ----------------------------
def build_sample_list(split: str, data_sources, seed=DATA_SAMPLE_SEED):
    rng = random.Random(seed if split == "train" else seed + 1)
    samples = []
    stats = {}
    for source in data_sources:
        ann_dir = Path(source["annotation_dir"])
        image_dir = Path(source["image_dir"])
        limit = source.get(f"{split}_limit")
        source_name = source.get("name", ann_dir.name)

        matched = []
        for fname in sorted(os.listdir(ann_dir)):
            if not fname.endswith(".json"):
                continue
            if not fname.endswith(f"_{split}.json"):
                # 保险起见读取sample_type
                with open(ann_dir / fname, "r", encoding="utf-8") as f:
                    ann = json.load(f)
                if ann.get("sample_type") != split:
                    continue
            matched.append(ann_dir / fname)

        available = len(matched)
        if limit is not None and available > limit:
            matched = rng.sample(matched, limit)

        stats[source_name] = {"available": available, "kept": len(matched), "limit": limit}
        for p in matched:
            samples.append({"ann_path": p, "image_dir": image_dir, "source": source_name})

    rng.shuffle(samples)
    if not samples:
        raise ValueError(f"{split} 样本为空，请检查数据源")
    return samples, stats


class MixedPDFDataset(Dataset):
    def __init__(self, samples, img_size=768, split="train"):
        self.samples = samples
        self.img_size = img_size
        self.split = split
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item_meta = self.samples[idx]
        ann_path = item_meta["ann_path"]
        image_dir = item_meta["image_dir"]

        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        img_path = image_dir / ann["overlay_image_name"]
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Load image {img_path} failed: {e}")
            image = torch.zeros(3, self.img_size, self.img_size)

        def get_bbox_xywh(element, key):
            if not isinstance(element, dict):
                return [0.0, 0.0, 0.0, 0.0]
            box = element.get(key)
            if not isinstance(box, dict):
                return [0.0, 0.0, 0.0, 0.0]
            return _xyxy_to_xywh(box.get('normalized_bbox', [0.0, 0.0, 0.0, 0.0]))

        this_elements = ann.get('this_elements', {})
        prev_elements = ann.get('prev_elements', {})
        next_elements = ann.get('next_elements', {})
        finger_info = ann.get('finger_info', {})
        la = ann.get('language_annotation', {})

        s_label = la.get('spatial_label', None)
        o_label = la.get('object_label', None)
        if s_label is None or o_label is None:
            ps, po = parse_labels_from_text(la.get('text_instruction', ''))
            s_label = s_label if s_label is not None else ps
            o_label = o_label if o_label is not None else po
        s_label = s_label if s_label in [0, 1, 2] else -1
        o_label = o_label if o_label in [0, 1, 2] else -1

        target = {
            'finger_bbox': torch.tensor(_xyxy_to_xywh(finger_info.get('normalized_bbox', [0.0]*4)), dtype=torch.float32),
            'this_word_bbox': torch.tensor(get_bbox_xywh(this_elements, 'word'), dtype=torch.float32),
            'this_line_bbox': torch.tensor(get_bbox_xywh(this_elements, 'line'), dtype=torch.float32),
            'this_paragraph_bbox': torch.tensor(get_bbox_xywh(this_elements, 'paragraph'), dtype=torch.float32),
            'prev_word_bbox': torch.tensor(get_bbox_xywh(prev_elements, 'word'), dtype=torch.float32),
            'prev_line_bbox': torch.tensor(get_bbox_xywh(prev_elements, 'line'), dtype=torch.float32),
            'prev_paragraph_bbox': torch.tensor(get_bbox_xywh(prev_elements, 'paragraph'), dtype=torch.float32),
            'next_word_bbox': torch.tensor(get_bbox_xywh(next_elements, 'word'), dtype=torch.float32),
            'next_line_bbox': torch.tensor(get_bbox_xywh(next_elements, 'line'), dtype=torch.float32),
            'next_paragraph_bbox': torch.tensor(get_bbox_xywh(next_elements, 'paragraph'), dtype=torch.float32),
            'spatial_label': torch.tensor(s_label, dtype=torch.long),
            'object_label': torch.tensor(o_label, dtype=torch.long),
            'instr_text': la.get('text_instruction', '')
        }

        return image, target, target['instr_text']


# ----------------------------
# 训练与验证
# ----------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, tokenizer):
    model.train()
    total_loss = 0.0
    metric_sum = {}

    iterable = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for images, targets, texts in iterable:
        images = images.to(device, non_blocking=True)
        for key in targets:
            if isinstance(targets[key], torch.Tensor):
                targets[key] = targets[key].to(device, non_blocking=True)

        token_batch = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=48,
            return_tensors="pt"
        )
        outputs = model(
            images,
            token_batch["input_ids"].to(device),
            token_batch.get("attention_mask").to(device) if token_batch.get("attention_mask") is not None else None,
            token_batch.get("token_type_ids").to(device) if token_batch.get("token_type_ids") is not None else None
        )
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['total_loss']

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for key, val in loss_dict.items():
            metric_sum.setdefault(key, 0.0)
            metric_sum[key] += float(val.item()) if torch.is_tensor(val) else float(val)

        postfix = {
            "Loss": f"{loss.item():.4f}",
            "IoU": f"{(1.0 - loss_dict['bbox_iou'].item()):.3f}",
            "SAcc": f"{loss_dict['spatial_acc'].item():.3f}",
            "OAcc": f"{loss_dict['object_acc'].item():.3f}",
        }
        ptr_iou = loss_dict.get('pointer_iou')
        if ptr_iou is not None:
            postfix["PtrIoU"] = f"{(1.0 - ptr_iou.item()):.3f}"
        iterable.set_postfix(postfix)

    num_batches = max(1, len(dataloader))
    mean_metrics = {k: v / num_batches for k, v in metric_sum.items()}
    mean_metrics['total_loss'] = total_loss / num_batches
    return mean_metrics


@torch.no_grad()
def validate(model, dataloader, criterion, device, tokenizer):
    model.eval()
    total_loss = 0.0
    metric_sum = {}

    for images, targets, texts in dataloader:
        images = images.to(device, non_blocking=True)
        for key in targets:
            if isinstance(targets[key], torch.Tensor):
                targets[key] = targets[key].to(device, non_blocking=True)

        token_batch = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=48,
            return_tensors="pt"
        )
        outputs = model(
            images,
            token_batch["input_ids"].to(device),
            token_batch.get("attention_mask").to(device) if token_batch.get("attention_mask") is not None else None,
            token_batch.get("token_type_ids").to(device) if token_batch.get("token_type_ids") is not None else None
        )
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['total_loss']

        total_loss += loss.item()
        for key, val in loss_dict.items():
            metric_sum.setdefault(key, 0.0)
            metric_sum[key] += float(val.item()) if torch.is_tensor(val) else float(val)

    num_batches = max(1, len(dataloader))
    mean_metrics = {k: v / num_batches for k, v in metric_sum.items()}
    mean_metrics['total_loss'] = total_loss / num_batches
    return mean_metrics


# ----------------------------
# 主训练入口
# ----------------------------
def main():
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    set_seed(42)

    # 构建采样列表
    train_samples, train_stats = build_sample_list("train", DATA_SOURCES, seed=DATA_SAMPLE_SEED)
    val_samples, val_stats = build_sample_list("test", DATA_SOURCES, seed=DATA_SAMPLE_SEED)

    # 数据集 & DataLoader
    train_dataset = MixedPDFDataset(train_samples, split="train")
    val_dataset = MixedPDFDataset(val_samples, split="test")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    print("数据加载完成：")
    print(f"  训练集：{len(train_dataset)} 样本 | 验证集：{len(val_dataset)} 样本")
    print(f"  train来源：" + "； ".join([f"{k}: {v['kept']}/{v['available']} (limit={v['limit'] or 'all'})" for k, v in train_stats.items()]))
    print(f"  val来源：" + "； ".join([f"{k}: {v['kept']}/{v['available']} (limit={v['limit'] or 'all'})" for k, v in val_stats.items()]))

    # 模型与优化器
    model = PDFWordLocator(
        bert_path="bert-base-chinese",
        device=device,
        min_iters=2,
        max_iters=5
    ).to(device)
    criterion = PDFLocatorLoss(loss_weights={
        "bbox_l1": 1.0,
        "bbox_iou": 1.0,
        "score": 1.0,
        "progress": 0.2,
        "spatial_cls": 1.0,
        "object_cls": 1.0,
        "pointer_l1": 1.0,
        "pointer_iou": 1.0,
    }, progress_weight=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float('inf')
    history = []
    epochs = 100

    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch+1}/{epochs} =====")
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, tokenizer)
        val_metrics = validate(model, val_loader, criterion, device, tokenizer)
        scheduler.step(val_metrics['total_loss'])

        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics
        })

        print(f"Train Loss: {train_metrics['total_loss']:.4f} | Val Loss: {val_metrics['total_loss']:.4f}")
        print(f"Val IoU: {(1.0 - val_metrics['bbox_iou']):.4f} | Val Score MSE: {val_metrics['score']:.4f}")
        print(f"Val Spatial Acc: {val_metrics.get('spatial_acc', 0.0):.4f} | Val Object Acc: {val_metrics.get('object_acc', 0.0):.4f}")
        print(f"Val Pointer IoU: {(1.0 - val_metrics.get('pointer_iou', 0.0)):.4f} | Pointer valid-ratio: {val_metrics.get('pointer_valid_ratio', 0.0):.4f}")
        print(f"Val valid-ratio: {val_metrics.get('valid_ratio', 0.0):.4f}")

        state_dict = model.state_dict()
        if val_metrics['total_loss'] < best_val:
            best_val = val_metrics['total_loss']
            ckpt_path = SAVE_DIR / f"best_model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics
            }, ckpt_path)
            print(f"Saved best model to {ckpt_path}")

    final_path = SAVE_DIR / "final_model.pth"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, final_path)
    print(f"\nTraining finished. Final model saved to {final_path}")
    print(f"Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
