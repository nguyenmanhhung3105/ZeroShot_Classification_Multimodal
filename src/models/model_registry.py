"""
Khai báo và tải model OpenCLIP dùng trong pipeline chính.

Mỗi model cung cấp embedding ảnh/text đã chuẩn hóa cùng scale và bias gốc.
Tokenizer nhận đúng context_length của model (SigLIP2: 64, CLIP-family: 77).
Runner quyết định softmax, sigmoid hoặc relative_z theo tác vụ và cấu hình;
wrapper chỉ tải và mã hóa, không huấn luyện hoặc tự chọn nhãn.
"""

import gc

import torch
import open_clip


# ============================================================
# CẤU HÌNH CÁC MODEL SẼ DÙNG TRONG DỰ ÁN
# ============================================================
# Phiên bản yêu cầu trong requirements.txt có đủ tên model/pretrained bên dưới.
MODEL_CONFIGS = {
    # Cùng checkpoint OpenAI CLIP ViT-B/32 dùng ở nhánh upstream; ở đây nạp
    # qua OpenCLIP để chạy cùng prompt, fusion và định dạng log của pipeline chính.
    "openai_clip_vit_b_32": {
        "arch": "ViT-B-32",
        "pretrained": "openai",
        "family": "clip_softmax",
    },
    # MetaCLIP2 dùng chung cho ba dataset; không mặc định kết luận tốt hơn model khác.
    "metaclip2_vith14": {
        "arch": "ViT-H-14-worldwide-quickgelu",   # giữ đúng hậu tố của kiến trúc pretrained
        "pretrained": "metaclip2_worldwide",
        "family": "clip_softmax",
    },
    "dfn5b_vith14": {
        "arch": "hf-hub:apple/DFN5B-CLIP-ViT-H-14",
        "pretrained": None,          # bắt buộc None khi arch là hf-hub:...
        "tokenizer": "ViT-H-14",     # Apple dùng tokenizer CLIP chuẩn, không dùng ID hf-hub
        "family": "clip_softmax",
    },

    # Dùng cho cả ba dataset; cách quyết định nhãn được chọn ở inference.py.
    "siglip2_so400m": {
        "arch": "ViT-SO400M-14-SigLIP2",
        "pretrained": "webli",
        "family": "siglip_sigmoid",
    },
}


class VLMWrapper:
    """
    Bọc model VLM (CLIP/SigLIP) thành giao diện thống nhất:
    - encode_images(list[PIL.Image]) -> tensor embedding đã chuẩn hóa
    - encode_texts(list[str]) -> tensor embedding đã chuẩn hóa
    - similarity(image_embeds, text_embeds) -> cosine thô để kiểm tra/ghi log
    - calibrated_logits(image_embeds, text_embeds) -> logits đã nhân logit_scale
      + cộng logit_bias, sẵn sàng để inference.py tự chọn softmax hay sigmoid
      tuỳ theo task (single-label hay multi-label), không phải tuỳ theo model.
    Dùng chung giao diện này để run_experiment.py không cần biết
    bên trong là CLIP hay SigLIP.
    """

    def __init__(self, model_name: str, device: str = None):
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Model '{model_name}' chưa được khai báo trong MODEL_CONFIGS. "
                f"Các model hợp lệ: {list(MODEL_CONFIGS.keys())}"
            )
        config = MODEL_CONFIGS[model_name]
        self.model_name = model_name
        self.family = config.get("family", "clip_softmax")
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"[model_registry] Đang load model '{model_name}' ({config['arch']}, "
              f"pretrained={config['pretrained']}) lên {self.device} ...")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config["arch"], pretrained=config["pretrained"]
        )
        tokenizer_id = config.get("tokenizer", config["arch"])
        self.tokenizer = open_clip.get_tokenizer(tokenizer_id)
        self.model.to(self.device)
        self.model.eval()  # QUAN TRỌNG: chế độ eval, tắt dropout/batchnorm update

        # Lấy tham số hiệu chỉnh đã học từ model, không gán giá trị cố định.
        # logit_bias chỉ tồn tại ở SigLIP-family; CLIP-family không có -> mặc định 0.
        self.logit_scale = float(self.model.logit_scale.exp().item())
        logit_bias = getattr(self.model, "logit_bias", None)
        self.logit_bias = float(logit_bias.item()) if logit_bias is not None else 0.0

        # Chiều embedding thực tế, tránh dùng nhầm bộ nhớ đệm của model khác.
        self.embed_dim = getattr(self.model.visual, "output_dim", None)

        # Kích thước ảnh lấy từ model thực tế, không dựa vào con số ghi tay trong tài liệu.
        actual_image_size = getattr(self.model.visual, "image_size", None)

        print(f"[model_registry] -> Load xong '{model_name}': "
              f"embed_dim={self.embed_dim}, image_size={actual_image_size}, "
              f"context_length={self.model.context_length}, "
              f"logit_scale={self.logit_scale:.2f}, logit_bias={self.logit_bias:.4f}")

    @torch.no_grad()
    def encode_images(self, images: list) -> torch.Tensor:
        """Nhận danh sách ảnh PIL chưa tiền xử lý."""
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        embeds = self.model.encode_image(batch)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds

    @torch.no_grad()
    def encode_texts(self, texts: list) -> torch.Tensor:
        """Nhận danh sách text hoặc prompt."""
        # Truyền context_length tường minh, đặc biệt SigLIP2 chỉ nhận 64 token.
        tokens = self.tokenizer(texts, context_length=self.model.context_length).to(self.device)
        embeds = self.model.encode_text(tokens)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds

    @staticmethod
    def similarity(image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """Cosine similarity thô [n_images, n_texts] — dùng để debug/log, KHÔNG dùng
        trực tiếp để fuse điểm giữa các model vì thang đo mỗi model khác nhau."""
        return image_embeds @ text_embeds.T

    def calibrated_logits(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """Logits đã hiệu chỉnh theo đúng logit_scale/logit_bias model đã học.
        inference.py dùng hàm này rồi tự quyết định:
        - single-label (Fakeddit/CrisisMMD) -> softmax(logits, dim=-1)
        - multi-label  (mmimdb)             -> sigmoid(logits)  [chỉ nên dùng
          cho siglip2_so400m — với clip_softmax, sigmoid không có ý nghĩa
          hiệu chỉnh vì logit_bias luôn = 0 và scale không được huấn luyện cho mục đích đó]
        """
        cos_sim = self.similarity(image_embeds, text_embeds)
        return cos_sim * self.logit_scale + self.logit_bias

    def unload(self):
        """Giải phóng bộ nhớ thiết bị trước khi tải model tiếp theo."""
        del self.model
        gc.collect()
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            print(f"[model_registry] VRAM sau giải phóng: "
                  f"allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB")
        elif str(self.device).startswith("mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        print(f"[model_registry] Đã giải phóng '{self.model_name}' khỏi bộ nhớ")


def load_model(model_name: str, device: str = None) -> VLMWrapper:
    """Hàm tải model dùng chung cho các điểm vào của dự án."""
    return VLMWrapper(model_name, device=device)


def list_available_models() -> list:
    return list(MODEL_CONFIGS.keys())
