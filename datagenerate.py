from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
import fitz  # PyMuPDF
import json
import os
import random
import re
from PIL import Image as PILImage, ImageDraw, ImageFont, ImageOps
import cv2


class PDFDatasetGenerator:
    def __init__(self, pdf_path, output_dir, target_size=(768, 768), min_samples=2000, mask_dir=None):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        self.target_size = target_size  # 768x768分辨率
        self.min_samples = min_samples  # 总样本数目标
        self.total_samples = 0  # 累计生成样本数
        self.generated_samples = set()  # 避免重复样本

        # 输出目录初始化
        self.image_dir = os.path.join(output_dir, "pawl_overlay_dataset")
        self.annotation_dir = os.path.join(output_dir, "pawl_coord_dataset")
        for dir_path in [self.image_dir, self.annotation_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 指示物掩码配置（适配768分辨率）
        self.mask_dir = mask_dir
        self.finger_mask_path = "finger.png"
        self.finger_tip_x, self.finger_tip_y = 45, 0  # 固定指尖坐标
        self.bg_color = None  # 自动识别背景色（采样四角）
        self.bg_tolerance = 30
        self.finger_global_scale = 5.2  # 单掩码时的默认放大系数
        self.mask_scale_jitter = (1.0, 1.0)  # 不使用随机扰动
        self.default_mask_ref_size = (3023, 4023)  # 默认原图尺寸参考
        self.mask_ref_size_overrides = {
            "(1673,1383).png": (3008, 4024)
        }
        self.tip_rect_size_by_mask = {
            "(938,1401).png": (342, 415),
            "(1005,1932).png": (319, 371),
            "(1688,1462).png": (293, 340),
            "(1673,1383).png": (187, 290),
        }
        self.mask_y_offset_ratio = 0.35  # 调整偏移比例适应新分辨率
        self.fixed_mask_base_width = 400  # 增大基础宽度
        self.partial_mask_coords = (0, 0, 106, 150)  # 手指部分框选区域

        # 字体加载
        try:
            self.font = ImageFont.truetype("Arial.ttf", 20)
        except:
            self.font = ImageFont.load_default()

        # 加载指示物掩码（支持多掩码随机抽选）
        self.mask_assets = []
        self.finger_mask = None
        if self.mask_dir:
            self.mask_assets = self.load_mask_assets(self.mask_dir)
            if not self.mask_assets:
                raise RuntimeError("指示物掩码加载失败，无法继续生成数据集")
        else:
            self.finger_mask = self.load_finger_mask()
            if not self.finger_mask:
                raise RuntimeError("手指掩码加载失败，无法继续生成数据集")

        # 数据集划分（默认8:2训练/验证）
        self.annotations = {
            "train": [],
            "test": []
        }

        # 语言指令模板（按空间关系×对象关系组织）
        # 空间关系: 0=this(这个), 1=prev(上一个), 2=next(下一个)
        # 对象关系: 0=word(单词), 1=line(行), 2=paragraph(段落) 【修改：sentence→line】
        # 每个类型约15个指令，仅包含单一对象
        self.language_templates = {
            (0, 0): [  # 这个 + 单词
                "帮我定位这个单词",
                "找到这个单词的位置",
                "输出这个单词的坐标",
                "标记这个单词的位置",
                "定位这个单词",
                "显示这个单词在哪里",
                "这个单词的位置是什么",
                "指出这个单词的位置",
                "找到这个单词",
                "确定这个单词的位置",
                "这个单词在什么地方",
                "标记出这个单词",
                "给出这个单词的坐标",
                "定位当前这个单词",
                "这个单词的位置在哪里"
            ],
            (0, 1): [  # 这个 + 行 【修改：句子→行】
                "帮我定位这一行",
                "找到这一行的位置",
                "输出这一行的坐标",
                "标记这一行的位置",
                "定位这一行",
                "显示这一行在哪里",
                "这一行的位置是什么",
                "指出这一行的位置",
                "找到这一行",
                "确定这一行的位置",
                "这一行在什么地方",
                "标记出这一行",
                "给出这一行的坐标",
                "定位当前这一行",
                "这一行的位置在哪里"
            ],
            (0, 2): [  # 这个 + 段落
                "帮我定位这个段落",
                "找到这个段落的位置",
                "输出这个段落的坐标",
                "标记这个段落的位置",
                "定位这个段落",
                "显示这个段落在哪里",
                "这个段落的位置是什么",
                "指出这个段落的位置",
                "找到这个段落",
                "确定这个段落的位置",
                "这个段落在什么地方",
                "标记出这个段落",
                "给出这个段落的坐标",
                "定位当前这个段落",
                "这个段落的位置在哪里"
            ],
            (1, 0): [  # 上一个 + 单词
                "帮我定位上一个单词",
                "找到上一个单词的位置",
                "输出上一个单词的坐标",
                "标记上一个单词的位置",
                "定位上一个单词",
                "显示上一个单词在哪里",
                "上一个单词的位置是什么",
                "指出上一个单词的位置",
                "找到上一个单词",
                "确定上一个单词的位置",
                "上一个单词在什么地方",
                "标记出上一个单词",
                "给出上一个单词的坐标",
                "上一个单词在哪里",
                "找到前面的那个单词"
            ],
            (1, 1): [  # 上一个 + 行 【修改：句子→行】
                "帮我定位上一行",
                "找到上一行的位置",
                "输出上一行的坐标",
                "标记上一行的位置",
                "定位上一行",
                "显示上一行在哪里",
                "上一行的位置是什么",
                "指出上一行的位置",
                "找到上一行",
                "确定上一行的位置",
                "上一行在什么地方",
                "标记出上一行",
                "给出上一行的坐标",
                "上一行在哪里",
                "找到前面的那一行"
            ],
            (1, 2): [  # 上一个 + 段落
                "帮我定位上一个段落",
                "找到上一个段落的位置",
                "输出上一个段落的坐标",
                "标记上一个段落的位置",
                "定位上一个段落",
                "显示上一个段落在哪里",
                "上一个段落的位置是什么",
                "指出上一个段落的位置",
                "找到上一个段落",
                "确定上一个段落的位置",
                "上一个段落在什么地方",
                "标记出上一个段落",
                "给出上一个段落的坐标",
                "上一个段落在哪里",
                "找到前面的那个段落"
            ],
            (2, 0): [  # 下一个 + 单词
                "帮我定位下一个单词",
                "找到下一个单词的位置",
                "输出下一个单词的坐标",
                "标记下一个单词的位置",
                "定位下一个单词",
                "显示下一个单词在哪里",
                "下一个单词的位置是什么",
                "指出下一个单词的位置",
                "找到下一个单词",
                "确定下一个单词的位置",
                "下一个单词在什么地方",
                "标记出下一个单词",
                "给出下一个单词的坐标",
                "下一个单词在哪里",
                "找到后面的那个单词"
            ],
            (2, 1): [  # 下一个 + 行 【修改：句子→行】
                "帮我定位下一行",
                "找到下一行的位置",
                "输出下一行的坐标",
                "标记下一行的位置",
                "定位下一行",
                "显示下一行在哪里",
                "下一行的位置是什么",
                "指出下一行的位置",
                "找到下一行",
                "确定下一行的位置",
                "下一行在什么地方",
                "标记出下一行",
                "给出下一行的坐标",
                "下一行在哪里",
                "找到后面的那一行"
            ],
            (2, 2): [  # 下一个 + 段落
                "帮我定位下一个段落",
                "找到下一个段落的位置",
                "输出下一个段落的坐标",
                "标记下一个段落的位置",
                "定位下一个段落",
                "显示下一个段落在哪里",
                "下一个段落的位置是什么",
                "指出下一个段落的位置",
                "找到下一个段落",
                "确定下一个段落的位置",
                "下一个段落在什么地方",
                "标记出下一个段落",
                "给出下一个段落的坐标",
                "下一个段落在哪里",
                "找到后面的那个段落"
            ]
        }
        
        # 映射关系：用于标签说明 【修改：sentence→line】
        self.spatial_mapping = {0: "this", 1: "prev", 2: "next"}
        self.object_mapping = {0: "word", 1: "line", 2: "paragraph"}  # 句子→行
        self.label_description = {
            (0,0): "这个单词",
            (0,1): "这一行",  # 这个句子→这一行
            (0,2): "这个段落",
            (1,0): "上一个单词",
            (1,1): "上一行",  # 上一个句子→上一行
            (1,2): "上一个段落",
            (2,0): "下一个单词",
            (2,1): "下一行",  # 下一个句子→下一行
            (2,2): "下一个段落"
        }


    def _guess_bg_color(self, img):
        """估计掩码背景色（采样四角取众数）"""
        width, height = img.size
        corners = [
            img.getpixel((0, 0)),
            img.getpixel((width - 1, 0)),
            img.getpixel((0, height - 1)),
            img.getpixel((width - 1, height - 1))
        ]
        corners = [c[:3] for c in corners]
        return max(corners, key=corners.count)


    def _get_mask_reference_size(self, mask_path):
        """根据掩码文件名获取原图尺寸参考"""
        basename = os.path.basename(mask_path)
        return self.mask_ref_size_overrides.get(basename, self.default_mask_ref_size)


    def _get_tip_rect_size(self, mask_path):
        """根据掩码文件名获取指尖下方矩形尺寸（宽, 高）"""
        basename = os.path.basename(mask_path)
        return self.tip_rect_size_by_mask.get(basename)


    def _parse_tip_from_filename(self, mask_path):
        """从文件名解析指示物尖端坐标，如 (1005,1932).png"""
        basename = os.path.basename(mask_path)
        match = re.search(r"\((-?\d+)\s*,\s*(-?\d+)\)", basename)
        if not match:
            raise ValueError(f"无法从掩码文件名解析指尖坐标: {basename}")
        return int(match.group(1)), int(match.group(2))


    def _compute_mask_bbox(self, mask_img):
        """计算掩码非透明区域边界框"""
        alpha = mask_img.split()[-1]
        bbox = alpha.getbbox()
        if bbox:
            return bbox
        return (0, 0, mask_img.width, mask_img.height)


    def load_mask_assets(self, mask_dir):
        """加载多张指示物掩码并解析尖端坐标"""
        if not os.path.isdir(mask_dir):
            print(f"❌ 掩码目录不存在: {mask_dir}")
            return []

        mask_paths = [
            os.path.join(mask_dir, name)
            for name in os.listdir(mask_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        mask_paths.sort()

        assets = []
        for mask_path in mask_paths:
            try:
                tip_x, tip_y = self._parse_tip_from_filename(mask_path)
            except ValueError as exc:
                print(f"⚠️  {exc}")
                continue

            mask_img = self.load_finger_mask(mask_path, verbose=False)
            if not mask_img:
                continue
            mask_bbox = self._compute_mask_bbox(mask_img)

            assets.append({
                "path": mask_path,
                "image": mask_img,
                "tip": (tip_x, tip_y),
                "bbox": mask_bbox
            })

        if assets:
            print(f"✅ 指示物掩码加载完成，共{len(assets)}个: {mask_dir}")
        else:
            print(f"❌ 指示物掩码加载失败: {mask_dir}")
        return assets


    def load_finger_mask(self, mask_path=None, verbose=True):
        """加载并预处理手指掩码（背景透明化）"""
        target_path = mask_path or self.finger_mask_path
        if not os.path.exists(target_path):
            print(f"❌ 手指掩码图片不存在: {target_path}")
            return None

        try:
            finger_mask = PILImage.open(target_path).convert("RGBA")
            width, height = finger_mask.size
            pixels = finger_mask.load()

            # 背景色透明化（匹配白色背景）
            bg_color = self.bg_color or self._guess_bg_color(finger_mask)
            for y in range(height):
                for x in range(width):
                    r, g, b, a = pixels[x, y]
                    if (abs(r - bg_color[0]) < self.bg_tolerance and
                            abs(g - bg_color[1]) < self.bg_tolerance and
                            abs(b - bg_color[2]) < self.bg_tolerance):
                        pixels[x, y] = (r, g, b, 0)

            if verbose:
                print(f"✅ 手指掩码加载完成（基础宽度：{self.fixed_mask_base_width}，放大系数：{self.finger_global_scale}）")
            return finger_mask
        except Exception as e:
            print(f"❌ 加载手指掩码失败: {e}")
            return None


    def _resize_to_target(self, img, return_params=False):
        """将图像缩放到768x768，保持宽高比+黑边填充"""
        original_width, original_height = img.size
        
        # 计算缩放比例（避免超出目标尺寸）
        scale = min(
            self.target_size[0] / original_width,
            self.target_size[1] / original_height
        )
        
        # 缩放后尺寸
        scaled_width = int(original_width * scale)
        scaled_height = int(original_height * scale)
        
        # 缩放图像
        scaled_img = img.resize((scaled_width, scaled_height), PILImage.LANCZOS)
        
        # 居中填充黑边（768分辨率专用计算）
        pad_left = (self.target_size[0] - scaled_width) // 2
        pad_top = (self.target_size[1] - scaled_height) // 2
        pad_right = self.target_size[0] - scaled_width - pad_left
        pad_bottom = self.target_size[1] - scaled_height - pad_top
        
        final_img = ImageOps.expand(
            scaled_img,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=(0, 0, 0)  # 黑边填充
        )
        
        if return_params:
            return final_img, scaled_width, scaled_height, pad_left, pad_top, scale
        else:
            return final_img


    def extract_elements_from_pdf(self, cycle_num=0):
        """提取PDF元素并生成样本"""
        doc = fitz.open(self.pdf_path)
        total_pages = len(doc)
        page_samples = 0  # 本轮生成的样本数

        for page_num in range(total_pages):
            # 若已达到目标样本数，提前退出
            if self.total_samples >= self.min_samples:
                break

            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]  # 文本块信息
            page_words = page.get_text("words")  # 单词信息：(x0,y0,x1,y1,text,...)

            # 1. 提取页面标题（过滤标题区域的元素）
            titles = self.extract_titles(page, blocks)

            # 2. 筛选有效单词（非标题区域+非空）并按阅读顺序排序
            valid_words = []
            for word in page_words:
                word_bbox = [word[0], word[1], word[2], word[3]]
                word_text = word[4].strip()
                # 排除标题内单词、空单词、过短单词（长度<1）
                if word_text and len(word_text) >= 1 and not self.is_word_in_title(word_bbox, titles):
                    valid_words.append({
                        "bbox": word_bbox,
                        "text": word_text,
                        "raw": word  # 保留原始数据
                    })

            # 无有效单词或有效单词数<3（无法提取上1/下1）则跳过
            if len(valid_words) < 3:
                print(f"⚠️  第{page_num + 1}页有效单词数不足3个（当前：{len(valid_words)}），跳过")
                continue

            # 按阅读顺序排序单词（先按y坐标，再按x坐标，符合PDF阅读习惯）
            valid_words_sorted = sorted(valid_words, key=lambda w: (w["bbox"][1], w["bbox"][0]))

            # 识别行首单词 + 提取所有行 【新增：提前提取行，复用行划分逻辑】
            line_threshold = 5.0  # 行间距阈值（像素）
            lines = []
            current_line = [valid_words_sorted[0]]
            
            for word in valid_words_sorted[1:]:
                # 比较与上一个单词的y坐标差
                y_diff = word["bbox"][1] - current_line[-1]["bbox"][1]
                if y_diff < line_threshold:
                    current_line.append(word)
                else:
                    lines.append(current_line)
                    current_line = [word]
            if current_line:
                lines.append(current_line)
            
            # 标记每个单词是否为行首 + 生成行数据（文本+边界框）
            line_data_list = []  # 存储所有行的信息：text、bbox、words
            for line in lines:
                if not line:
                    continue
                # 标记行首
                line[0]["is_first_in_row"] = True
                for word in line[1:]:
                    word["is_first_in_row"] = False
                # 生成行文本
                line_text = " ".join([word["text"] for word in line])
                # 生成行边界框（最小x0、y0，最大x1、y1）
                line_x0 = min(word["bbox"][0] for word in line)
                line_y0 = min(word["bbox"][1] for word in line)
                line_x1 = max(word["bbox"][2] for word in line)
                line_y1 = max(word["bbox"][3] for word in line)
                # 保存行数据
                line_data_list.append({
                    "line_text": line_text,
                    "bbox": [line_x0, line_y0, line_x1, line_y1],
                    "words": line  # 该行包含的所有单词
                })

            # 3. 随机选1个单词作为核心（需保证有上1和下1单词）
            # 排除第一个和最后一个单词，确保能提取上1/下1
            candidate_indices = range(1, len(valid_words_sorted) - 1)
            if not candidate_indices:
                print(f"⚠️  第{page_num + 1}页无法找到满足条件的核心单词（需前后各1个单词），跳过")
                continue
            
            selected_idx = random.choice(candidate_indices)
            selected_word = valid_words_sorted[selected_idx]
            word_text = selected_word["text"].lower()  # 统一小写用于匹配
            word_bbox = selected_word["bbox"]
            
            # 获取当前单词是否为行首的标识
            is_first_in_row = selected_word.get("is_first_in_row", False)

            # 提取相邻单词（上1/下1）
            adjacent_words = {
                "prev": valid_words_sorted[selected_idx - 1],  # 上1个单词
                "next": valid_words_sorted[selected_idx + 1]   # 下1个单词
            }

            # 4. 提取包含该单词的行（位置验证）【修改：句子→行】
            containing_lines = []
            for line_data in line_data_list:
                # 位置验证：单词边界框必须完全在行边界框内
                line_bbox = line_data["bbox"]
                pos_contains = self.is_bbox_inside(selected_word["bbox"], line_bbox)
                
                if pos_contains:
                    containing_lines.append(line_data)

            # 无符合条件的行则跳过【修改：句子→行】
            if not containing_lines:
                print(f"⚠️  第{page_num + 1}页未找到包含单词「{word_text}」的行，跳过")
                continue
            
            containing_line = random.choice(containing_lines)  # 包含核心单词的行
            
            # 5. 找到当前行的上一个和下一个行【修改：句子→行】
            line_index = line_data_list.index(containing_line)
            prev_line = line_data_list[line_index - 1] if line_index > 0 else None  # 上一行
            next_line = line_data_list[line_index + 1] if line_index < len(line_data_list) - 1 else None  # 下一行

            # 6. 提取包含该行的段落（文本+位置验证）【修改：句子→行】
            paragraphs = self.extract_paragraphs(page, blocks, titles)
            containing_paragraphs = []
            
            for para in paragraphs:
                para_text_lower = para["text"].lower()
                line_text_lower = containing_line["line_text"].lower()
                
                # 文本验证：段落必须包含行的大部分内容（至少50%）
                line_words = set(line_text_lower.split())
                para_words = set(para_text_lower.split())
                overlap_ratio = len(line_words & para_words) / len(line_words) if line_words else 0
                text_contains = overlap_ratio > 0.5
                
                # 位置验证：行边界框必须完全在段落边界框内
                line_bbox = containing_line["bbox"]
                pos_contains = self.is_bbox_inside(line_bbox, para["bbox"])
                
                if text_contains and pos_contains:
                    containing_paragraphs.append(para)

            # 无符合条件的段落则跳过
            if not containing_paragraphs:
                print(f"⚠️  第{page_num + 1}页未找到包含行的段落，跳过")
                continue
            
            containing_paragraph = random.choice(containing_paragraphs)
            
            # 7. 找到当前段落的上一个和下一个段落
            para_index = paragraphs.index(containing_paragraph)
            prev_paragraph = paragraphs[para_index - 1] if para_index > 0 else None
            next_paragraph = paragraphs[para_index + 1] if para_index < len(paragraphs) - 1 else None

            # 8. 生成样本唯一标识（包含相邻单词信息，避免循环重复）
            prev_text = adjacent_words["prev"]["text"][:5]
            next_text = adjacent_words["next"]["text"][:5]
            sample_unique_id = f"page_{page_num+1}_word_{word_text[:10]}_prev_{prev_text}_next_{next_text}_cycle_{cycle_num}"
            if sample_unique_id in self.generated_samples:
                continue  # 跳过重复样本
            self.generated_samples.add(sample_unique_id)

            # 9. 渲染页面图像并添加手指掩码（不绘制任何边界框）
            dpi = 300
            scale_ratio = dpi / 72  # PDF默认72DPI，缩放至300DPI
            mat = fitz.Matrix(scale_ratio, scale_ratio)
            page_pix = page.get_pixmap(matrix=mat)
            page_img = PILImage.frombytes("RGB", [page_pix.width, page_pix.height], page_pix.samples).convert("RGBA")

            # 计算单词在渲染图中的边界框
            scaled_word_bbox = [coord * scale_ratio for coord in word_bbox]

            # 添加手指掩码（固定在单词下方）
            page_img_with_finger, mask_tip_coords, finger_bbox = self.add_finger_mask(
                page_img.copy(), scaled_word_bbox, scale_ratio
            )

            # 10. 图像缩放至768x768并获取参数
            final_img, _, _, pad_left, pad_top, scale_to_768 = self._resize_to_target(
                page_img_with_finger, return_params=True
            )

            # 11. 计算所有坐标的归一化值
            # 11.1 基础元素归一化（指尖、当前单词、行、段落）【修改：句子→行】
            # 指尖坐标归一化
            mask_tip_x, mask_tip_y = mask_tip_coords
            final_tip_x = (mask_tip_x * scale_to_768) + pad_left
            final_tip_y = (mask_tip_y * scale_to_768) + pad_top
            normalized_tip = [
                max(0, min(1, final_tip_x / self.target_size[0])),
                max(0, min(1, final_tip_y / self.target_size[1]))
            ]

            # 当前单词边界框归一化
            normalized_word_bbox = [
                (scaled_word_bbox[0] * scale_to_768 + pad_left) / self.target_size[0],
                (scaled_word_bbox[1] * scale_to_768 + pad_top) / self.target_size[1],
                (scaled_word_bbox[2] * scale_to_768 + pad_left) / self.target_size[0],
                (scaled_word_bbox[3] * scale_to_768 + pad_top) / self.target_size[1]
            ]

            # 当前行边界框归一化【修改：句子→行】
            scaled_line_bbox = [coord * scale_ratio for coord in containing_line["bbox"]]
            normalized_line_bbox = [
                (scaled_line_bbox[0] * scale_to_768 + pad_left) / self.target_size[0],
                (scaled_line_bbox[1] * scale_to_768 + pad_top) / self.target_size[1],
                (scaled_line_bbox[2] * scale_to_768 + pad_left) / self.target_size[0],
                (scaled_line_bbox[3] * scale_to_768 + pad_top) / self.target_size[1]
            ]

            # 段落边界框归一化
            scaled_para_bbox = [coord * scale_ratio for coord in containing_paragraph["bbox"]]
            normalized_para_bbox = [
                (scaled_para_bbox[0] * scale_to_768 + pad_left) / self.target_size[0],
                (scaled_para_bbox[1] * scale_to_768 + pad_top) / self.target_size[1],
                (scaled_para_bbox[2] * scale_to_768 + pad_left) / self.target_size[0],
                (scaled_para_bbox[3] * scale_to_768 + pad_top) / self.target_size[1]
            ]

            # 手指边界框归一化
            normalized_finger_bbox = [
                (finger_bbox[0] * scale_to_768 + pad_left) / self.target_size[0],
                (finger_bbox[1] * scale_to_768 + pad_top) / self.target_size[1],
                (finger_bbox[2] * scale_to_768 + pad_left) / self.target_size[0],
                (finger_bbox[3] * scale_to_768 + pad_top) / self.target_size[1]
            ]

            # 11.2 相邻单词、行、段落边界框归一化【修改：句子→行】
            normalized_prev_bbox = {}
            normalized_next_bbox = {}
            
            # 上一个单词
            scaled_prev_word_bbox = [coord * scale_ratio for coord in adjacent_words["prev"]["bbox"]]
            normalized_prev_bbox["word"] = [
                (scaled_prev_word_bbox[0] * scale_to_768 + pad_left) / self.target_size[0],
                (scaled_prev_word_bbox[1] * scale_to_768 + pad_top) / self.target_size[1],
                (scaled_prev_word_bbox[2] * scale_to_768 + pad_left) / self.target_size[0],
                (scaled_prev_word_bbox[3] * scale_to_768 + pad_top) / self.target_size[1]
            ]
            
            # 下一个单词
            scaled_next_word_bbox = [coord * scale_ratio for coord in adjacent_words["next"]["bbox"]]
            normalized_next_bbox["word"] = [
                (scaled_next_word_bbox[0] * scale_to_768 + pad_left) / self.target_size[0],
                (scaled_next_word_bbox[1] * scale_to_768 + pad_top) / self.target_size[1],
                (scaled_next_word_bbox[2] * scale_to_768 + pad_left) / self.target_size[0],
                (scaled_next_word_bbox[3] * scale_to_768 + pad_top) / self.target_size[1]
            ]
            
            # 上一行【修改：上一个句子→上一行】
            if prev_line:
                scaled_prev_line_bbox = [coord * scale_ratio for coord in prev_line["bbox"]]
                normalized_prev_bbox["line"] = [
                    (scaled_prev_line_bbox[0] * scale_to_768 + pad_left) / self.target_size[0],
                    (scaled_prev_line_bbox[1] * scale_to_768 + pad_top) / self.target_size[1],
                    (scaled_prev_line_bbox[2] * scale_to_768 + pad_left) / self.target_size[0],
                    (scaled_prev_line_bbox[3] * scale_to_768 + pad_top) / self.target_size[1]
                ]
            
            # 下一行【修改：下一个句子→下一行】
            if next_line:
                scaled_next_line_bbox = [coord * scale_ratio for coord in next_line["bbox"]]
                normalized_next_bbox["line"] = [
                    (scaled_next_line_bbox[0] * scale_to_768 + pad_left) / self.target_size[0],
                    (scaled_next_line_bbox[1] * scale_to_768 + pad_top) / self.target_size[1],
                    (scaled_next_line_bbox[2] * scale_to_768 + pad_left) / self.target_size[0],
                    (scaled_next_line_bbox[3] * scale_to_768 + pad_top) / self.target_size[1]
                ]
            
            # 上一个段落
            if prev_paragraph:
                scaled_prev_para_bbox = [coord * scale_ratio for coord in prev_paragraph["bbox"]]
                normalized_prev_bbox["paragraph"] = [
                    (scaled_prev_para_bbox[0] * scale_to_768 + pad_left) / self.target_size[0],
                    (scaled_prev_para_bbox[1] * scale_to_768 + pad_top) / self.target_size[1],
                    (scaled_prev_para_bbox[2] * scale_to_768 + pad_left) / self.target_size[0],
                    (scaled_prev_para_bbox[3] * scale_to_768 + pad_top) / self.target_size[1]
                ]
            
            # 下一个段落
            if next_paragraph:
                scaled_next_para_bbox = [coord * scale_ratio for coord in next_paragraph["bbox"]]
                normalized_next_bbox["paragraph"] = [
                    (scaled_next_para_bbox[0] * scale_to_768 + pad_left) / self.target_size[0],
                    (scaled_next_para_bbox[1] * scale_to_768 + pad_top) / self.target_size[1],
                    (scaled_next_para_bbox[2] * scale_to_768 + pad_left) / self.target_size[0],
                    (scaled_next_para_bbox[3] * scale_to_768 + pad_top) / self.target_size[1]
                ]

            # 12. 划分训练/验证集
            sample_type = "train" if random.random() < 0.8 else "test"

            # 13. 生成语言标注（基于空间和对象关系的组合）【修改：句子→行】
            # 先检查可用的标签组合
            available_labels = []
            # 当前系列（this）- 总是可用
            available_labels.append((0, 0))  # 这个单词
            available_labels.append((0, 1))  # 这一行（原：这个句子）
            available_labels.append((0, 2))  # 这个段落
            
            # 上一个系列（prev）- 仅当对应元素存在时可用
            available_labels.append((1, 0))  # 上一个单词（总是存在，因为已过滤）
            if prev_line:
                available_labels.append((1, 1))  # 上一行（原：上一个句子）
            if prev_paragraph:
                available_labels.append((1, 2))  # 上一个段落
            
            # 下一个系列（next）- 仅当对应元素存在时可用
            available_labels.append((2, 0))  # 下一个单词（总是存在，因为已过滤）
            if next_line:
                available_labels.append((2, 1))  # 下一行（原：下一个句子）
            if next_paragraph:
                available_labels.append((2, 2))  # 下一个段落
            
            # 如果没有可用标签，跳过该页
            if not available_labels:
                print(f"⚠️  第{page_num + 1}页没有可用的标签，跳过")
                continue
                
            # 随机选择一个可用标签组合
            selected_label = random.choice(available_labels)
            selected_instruction = random.choice(self.language_templates[selected_label])
            spatial_label, object_label = selected_label

            # 14. 保存图像（不含任何矩形框）
            image_filename = f"{self.pdf_name}_{sample_unique_id}_{sample_type}.png"
            image_path = os.path.join(self.image_dir, image_filename)
            final_img.convert("RGB").save(image_path, dpi=(dpi, dpi))

            # 15. 保存标注（使用新的结构：this_elements, prev_elements, next_elements）【修改：sentence→line】
            annotation = {
                "sample_id": sample_unique_id,
                "sample_type": sample_type,  # train/test
                "overlay_image_name": image_filename,
                "finger_info": {
                    "normalized_bbox": normalized_finger_bbox
                },
                "this_elements": {  # 这个系列
                    "word": {
                        "text": selected_word["text"],
                        "normalized_bbox": normalized_word_bbox,
                        "is_first_in_row": is_first_in_row
                    },
                    "line": {  # 原：sentence→line
                        "text": containing_line["line_text"],
                        "normalized_bbox": normalized_line_bbox
                    },
                    "paragraph": {
                        "text": containing_paragraph["text"],
                        "normalized_bbox": normalized_para_bbox
                    }
                },
                "prev_elements": {  # 上一个系列
                    "word": {
                        "text": adjacent_words["prev"]["text"],
                        "normalized_bbox": normalized_prev_bbox["word"],
                        "is_first_in_row": adjacent_words["prev"].get("is_first_in_row", False)
                    },
                    "line": {  # 原：sentence→line
                        "text": prev_line["line_text"] if prev_line else "",
                        "normalized_bbox": normalized_prev_bbox.get("line", [0,0,0,0])
                    } if prev_line else None,
                    "paragraph": {
                        "text": prev_paragraph["text"] if prev_paragraph else "",
                        "normalized_bbox": normalized_prev_bbox.get("paragraph", [0,0,0,0])
                    } if prev_paragraph else None
                },
                "next_elements": {  # 下一个系列
                    "word": {
                        "text": adjacent_words["next"]["text"],
                        "normalized_bbox": normalized_next_bbox["word"],
                        "is_first_in_row": adjacent_words["next"].get("is_first_in_row", False)
                    },
                    "line": {  # 原：sentence→line
                        "text": next_line["line_text"] if next_line else "",
                        "normalized_bbox": normalized_next_bbox.get("line", [0,0,0,0])
                    } if next_line else None,
                    "paragraph": {
                        "text": next_paragraph["text"] if next_paragraph else "",
                        "normalized_bbox": normalized_next_bbox.get("paragraph", [0,0,0,0])
                    } if next_paragraph else None
                },
                "language_annotation": {
                    "text_instruction": selected_instruction,  # 语言指令（已替换为行）
                    "spatial_label": spatial_label,  # 空间关系标签：0=this,1=prev,2=next
                    "object_label": object_label     # 对象关系标签：0=word,1=line,2=paragraph（已修改）
                },
                "marker_info": {  # 指尖信息
                    "normalized_tip_coord": normalized_tip,
                    "width": random.uniform(0.02, 0.05),
                    "tip_rel": [random.uniform(0.3, 0.7), random.uniform(0.3, 0.7)]
                },
                "image_info": {
                    "width": self.target_size[0],
                    "height": self.target_size[1]
                }
            }

            # 保存标注文件
            annotation_filename = f"{self.pdf_name}_{sample_unique_id}_{sample_type}.json"
            annotation_path = os.path.join(self.annotation_dir, annotation_filename)
            with open(annotation_path, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)

            # 更新计数和标注列表
            self.annotations[sample_type].append(annotation)
            self.total_samples += 1
            page_samples += 1

            # 打印进度（显示行首标识）
            row_status = "行首单词" if is_first_in_row else "非行首单词"
            print(f"✅ 第{page_num + 1}页生成样本（累计：{self.total_samples}/{self.min_samples}）| {row_status} | 标签：({spatial_label},{object_label})={self.label_description[selected_label]} | 核心单词：{word_text}")

        doc.close()
        return page_samples


    # ----------------------------
    # 辅助函数（删除原句子提取函数，无需保留）
    # ----------------------------
    def is_bbox_inside(self, inner_bbox, outer_bbox, tolerance=2.0):
        """检查内部边界框是否完全在外部边界框内（带容差）"""
        x0_in, y0_in, x1_in, y1_in = inner_bbox
        x0_out, y0_out, x1_out, y1_out = outer_bbox
        
        return (x0_in >= x0_out - tolerance and
                x1_in <= x1_out + tolerance and
                y0_in >= y0_out - tolerance and
                y1_in <= y1_out + tolerance)


    def extract_titles(self, page, blocks):
        """提取页面标题"""
        all_font_sizes = []
        text_blocks = []

        for block in blocks:
            if block["type"] == 0:  # 仅处理文本块
                block_info = {
                    "text": "",
                    "max_font": 0,
                    "bbox": block["bbox"]
                }
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_info["text"] += span["text"] + " "
                        block_info["max_font"] = max(block_info["max_font"], span["size"])
                        all_font_sizes.append(span["size"])
                block_info["text"] = block_info["text"].strip()
                text_blocks.append(block_info)

        # 计算标题字体阈值（75%分位+最小阈值）
        if all_font_sizes:
            all_font_sizes_sorted = sorted(all_font_sizes)
            percentile_75 = all_font_sizes_sorted[int(len(all_font_sizes_sorted) * 0.75)]
            title_font_threshold = max(percentile_75 * 1.4, 14)
        else:
            title_font_threshold = 14

        titles = []
        for block in text_blocks:
            block_text = block["text"]
            block_max_font = block["max_font"]
            block_bbox = block["bbox"]

            if not block_text:
                continue

            # 多条件判断是否为标题
            is_title = False
            # 条件1：字体大且文本短
            if block_max_font > title_font_threshold and len(block_text) < 200:
                is_title = True
            # 条件2：标题格式（如第X章、1.1.等）
            elif not is_title and len(block_text) < 150:
                title_pattern = re.match(
                    r'^(第[一二三四五六七八九十零百千0-9]+[章节条款项目节])|'
                    r'^([0-9]+\.[0-9]+(\.| |$))|'
                    r'^([IVXLCDM]+\s*\.?)|'
                    r'^([A-Za-z]+\s*\.?)|'
                    r'^([0-9]+\s*[、.)])|'
                    r'^(\(\s*[0-9]+\s*\))',
                    block_text.strip()
                )
                if title_pattern:
                    is_title = True
            # 条件3：常见章节标题（如摘要、引言等）
            elif not is_title:
                section_titles = ['摘要', '关键词', '引言', '背景', '方法', '实验', '结果', '讨论', '结论', '参考文献']
                for st in section_titles:
                    if block_text.startswith(st) and len(block_text) < 100:
                        is_title = True
                        break

            if is_title:
                titles.append({
                    "page": page.number + 1,
                    "text": block_text,
                    "bbox": block_bbox,
                    "font_size": block_max_font
                })

        return titles


    def extract_paragraphs(self, page, blocks, titles):
        """提取非标题段落"""
        paragraphs = []
        for block in blocks:
            if block["type"] == 0:  # 文本块
                block_info = {
                    "text": "",
                    "max_font": 0,
                    "bbox": block["bbox"]
                }
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_info["text"] += span["text"] + " "
                        block_info["max_font"] = max(block_info["max_font"], span["size"])
                block_info["text"] = block_info["text"].strip()

                if not block_info["text"]:
                    continue

                # 过滤：不与标题重叠+不包含标题文本
                if (not self.is_para_overlapping_title(block_info, titles) and
                        not self.para_contains_title_text(block_info, titles)):
                    paragraphs.append(block_info)
        return paragraphs


    def add_finger_mask(self, image, elem_bbox, scale_ratio):
        """添加手指掩码（固定在单词下方，适配768分辨率）"""
        if self.mask_assets:
            mask_asset = random.choice(self.mask_assets)
            mask_img = mask_asset["image"]
            tip_x, tip_y = mask_asset["tip"]
            tip_rect_size = self._get_tip_rect_size(mask_asset["path"])
            if tip_rect_size:
                rect_w, rect_h = tip_rect_size
                rect_x0 = tip_x - rect_w / 2
                rect_y0 = tip_y
                rect_x1 = rect_x0 + rect_w
                rect_y1 = rect_y0 + rect_h
                rect_x0 = max(0, min(mask_img.width, rect_x0))
                rect_y0 = max(0, min(mask_img.height, rect_y0))
                rect_x1 = max(rect_x0, min(mask_img.width, rect_x1))
                rect_y1 = max(rect_y0, min(mask_img.height, rect_y1))
                mask_bbox = (rect_x0, rect_y0, rect_x1, rect_y1)
            else:
                mask_bbox = mask_asset["bbox"]
        else:
            mask_img = self.finger_mask
            tip_x, tip_y = self.finger_tip_x, self.finger_tip_y
            mask_bbox = self.partial_mask_coords

        # 单词中心点
        elem_center_x = elem_bbox[0] + (elem_bbox[2] - elem_bbox[0]) / 2
        elem_center_y = elem_bbox[1] + (elem_bbox[3] - elem_bbox[1]) / 2

        # 计算掩码尺寸（按原图比例缩放+随机扰动）
        if self.mask_assets:
            ref_w, ref_h = self._get_mask_reference_size(mask_asset["path"])
            scale_w = image.width / ref_w
            scale_h = image.height / ref_h
            base_scale = min(scale_w, scale_h)
            jitter = random.uniform(*self.mask_scale_jitter)
            scale = base_scale * jitter
            final_mask_width = max(1, int(mask_img.width * scale))
            final_mask_height = max(1, int(mask_img.height * scale))
        else:
            base_mask_width = self.fixed_mask_base_width
            final_mask_width = int(base_mask_width * self.finger_global_scale)
            original_mask_ratio = mask_img.height / mask_img.width
            final_mask_height = int(final_mask_width * original_mask_ratio)
        
        # 缩放掩码
        resized_mask = mask_img.resize((final_mask_width, final_mask_height), PILImage.LANCZOS)

        # 计算指尖位置（基于固定坐标缩放）
        tip_scale = final_mask_width / mask_img.width
        adjusted_tip_x = tip_x * tip_scale
        adjusted_tip_y = tip_y * tip_scale

        # 调整掩码位置（放在单词下方，偏移比例控制）
        mask_x = int(elem_center_x - adjusted_tip_x)
        mask_y = int(elem_center_y - adjusted_tip_y + (elem_bbox[3] - elem_bbox[1]) * self.mask_y_offset_ratio)

        # 计算掩码尖端在图像中的实际坐标
        mask_tip_x = mask_x + adjusted_tip_x
        mask_tip_y = mask_y + adjusted_tip_y
        
        # 计算手指部分区域的边界框
        scale_x = final_mask_width / mask_img.width
        scale_y = final_mask_height / mask_img.height
        partial_x0, partial_y0, partial_x1, partial_y1 = mask_bbox
        finger_bbox = (
            mask_x + int(partial_x0 * scale_x),
            mask_y + int(partial_y0 * scale_y),
            mask_x + int(partial_x1 * scale_x),
            mask_y + int(partial_y1 * scale_y)
        )

        # 叠加掩码（透明合成）
        mask_layer = PILImage.new("RGBA", image.size, (0, 0, 0, 0))
        mask_layer.paste(resized_mask, (mask_x, mask_y), resized_mask)
        combined_img = PILImage.alpha_composite(image, mask_layer)
        
        return combined_img, (mask_tip_x, mask_tip_y), finger_bbox


    def para_contains_title_text(self, para, titles):
        """检查段落是否包含标题文本"""
        para_text = para["text"].lower()
        for title in titles:
            title_text = title["text"].lower()
            if len(title_text) > 3 and title_text in para_text:
                return True
        return False


    def is_para_overlapping_title(self, para, titles):
        """检查段落是否与标题重叠"""
        para_x0, para_y0, para_x1, para_y1 = para["bbox"]
        for title in titles:
            title_x0, title_y0, title_x1, title_y1 = title["bbox"]
            overlap = (para_x0 < title_x1 - 1 and
                       para_x1 > title_x0 + 1 and
                       para_y0 < title_y1 - 1 and
                       para_y1 > title_y0 + 1)
            if overlap:
                return True
        return False


    def is_word_in_title(self, word_bbox, titles):
        """检查单词是否在标题区域内"""
        word_x0, word_y0, word_x1, word_y1 = word_bbox
        for title in titles:
            title_x0, title_y0, title_x1, title_y1 = title["bbox"]
            overlap = (word_x0 < title_x1 and
                       word_x1 > title_x0 and
                       word_y0 < title_y1 and
                       word_y1 > title_y0)
            if overlap:
                return True
        return False


    def save_dataset_summary(self):
        """保存数据集摘要"""
        train_count = len(self.annotations["train"])
        test_count = len(self.annotations["test"])

        # 统计各标签组合样本数
        label_count = {}
        for sl in range(3):
            for ol in range(3):
                label_count[(sl, ol)] = 0
        
        # 统计行首/非行首单词数量
        row_status_count = {
            "first_in_row": 0,
            "non_first_in_row": 0
        }
        
        for sample_type in ["train", "test"]:
            for ann in self.annotations[sample_type]:
                sl = ann["language_annotation"]["spatial_label"]
                ol = ann["language_annotation"]["object_label"]
                label_count[(sl, ol)] += 1
                
                # 统计行首/非行首单词
                if ann["this_elements"]["word"]["is_first_in_row"]:
                    row_status_count["first_in_row"] += 1
                else:
                    row_status_count["non_first_in_row"] += 1

        summary = {
            "pdf_name": self.pdf_name,
            "target_samples": self.min_samples,
            "generated_samples": self.total_samples,
            "train_test_split": "8:2",
            "train_samples": train_count,
            "test_samples": test_count,
            "image_size": f"{self.target_size[0]}x{self.target_size[1]}",
            "row_status_distribution": {
                "行首单词": row_status_count["first_in_row"],
                "非行首单词": row_status_count["non_first_in_row"],
                "比例": f"{row_status_count['first_in_row']/max(1,row_status_count['non_first_in_row']):.2f}"
            },
            "language_annotation_summary": {
                "spatial_mapping": "0=this(这个), 1=prev(上一个), 2=next(下一个)",
                "object_mapping": "0=word(单词), 1=line(行), 2=paragraph(段落)",  # 修改：sentence→line
                "label_distribution": {
                    f"{self.label_description[(0,0)]}(0,0)": label_count[(0,0)],
                    f"{self.label_description[(0,1)]}(0,1)": label_count[(0,1)],  # 这一行
                    f"{self.label_description[(0,2)]}(0,2)": label_count[(0,2)],
                    f"{self.label_description[(1,0)]}(1,0)": label_count[(1,0)],
                    f"{self.label_description[(1,1)]}(1,1)": label_count[(1,1)],  # 上一行
                    f"{self.label_description[(1,2)]}(1,2)": label_count[(1,2)],
                    f"{self.label_description[(2,0)]}(2,0)": label_count[(2,0)],
                    f"{self.label_description[(2,1)]}(2,1)": label_count[(2,1)],  # 下一行
                    f"{self.label_description[(2,2)]}(2,2)": label_count[(2,2)]
                },
                "instruction_templates": f"每个标签组合对应{len(self.language_templates[(0,0)])}条多样化指令"
            },
            "sample_structure": {
                "core": "选中的非标题单词（前后各1个单词，确保相邻单词有效）",
                "elements": "this_elements(当前)、prev_elements(上一个)、next_elements(下一个)各包含word/line/paragraph",  # 修改：sentence→line
                "validation": "单词在该行内、行在段落内、相邻元素按阅读顺序排序",  # 修改：句子→行
                "image_content": "仅包含原始PDF内容和手指掩码，无任何边界框绘制"
            }
        }

        summary_path = os.path.join(self.output_dir, f"{self.pdf_name}_dataset_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 打印最终统计
        print(f"\n📊 数据集摘要已保存: {summary_path}")
        print(f"总样本数: {self.total_samples}/{self.min_samples}")
        print(f"训练集: {train_count} 个 | 验证集: {test_count} 个")
        print(f"行首单词: {row_status_count['first_in_row']} 个 | 非行首单词: {row_status_count['non_first_in_row']} 个")
        print("标签分布:")
        for (sl, ol), desc in self.label_description.items():
            print(f"  {desc}({sl},{ol}): {label_count[(sl, ol)]} 个")


    def generate_dataset(self):
        """主生成逻辑"""
        print(f"🚀 开始处理PDF: {self.pdf_path}")
        print(f"目标样本数: {self.min_samples} | 图像尺寸: {self.target_size[0]}x{self.target_size[1]}")
        print(f"样本规则: 每页1个样本（需前后各1个单词），仅含原始内容+手指掩码，严格验证层级关系（单词→行→段落）\n")  # 修改：句子→行

        cycle_num = 0  # 循环轮次（避免重复）
        while self.total_samples < self.min_samples:
            print(f"----- 第{cycle_num + 1}轮处理 -----")
            new_samples = self.extract_elements_from_pdf(cycle_num)
            
            # 若本轮未生成任何样本，说明PDF无更多可用内容
            if new_samples == 0:
                print("⚠️  本轮未生成任何样本，PDF可能已无有效内容，终止生成")
                break
            
            cycle_num += 1

        # 保存摘要并打印结果
        self.save_dataset_summary()
        print(f"\n✅ 数据集生成完成!")
        print(f"图像路径: {self.image_dir}")
        print(f"标注路径: {self.annotation_dir}")


# 使用示例
if __name__ == "__main__":
    # 检查并安装依赖
    required_packages = ["fitz", "PyMuPDF", "Pillow", "opencv-python", "json5"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"⚠️  正在安装依赖库 {package}...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} 安装完成")

    # 请根据实际路径修改
    pdf_path = "./word.pdf"  # 你的PDF文件路径
    output_dir = "./pdf_dataset_enhanced"  # 输出目录

    # 初始化生成器（目标样本数，768x768分辨率）
    mask_dir = "./mask"

    generator = PDFDatasetGenerator(
        pdf_path=pdf_path,
        output_dir=output_dir,
        target_size=(768, 768),  # 768x768分辨率
        min_samples=15000,
        mask_dir=mask_dir
    )

    # 开始生成
    generator.generate_dataset()
