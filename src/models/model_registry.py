"""
Model Registry — nơi khai báo và load các Vision-Language Model dùng cho zero-shot.

NGUYÊN TẮC QUAN TRỌNG (đã thống nhất trong quá trình thiết kế dự án):
- Mỗi model chỉ được load ĐÚNG 1 LẦN, tái sử dụng xuyên suốt nhiều dataset và nhiều prompt set.
- Đây là inference thuần túy (forward pass), không có bước train/fine-tune nào —
  vì vậy không có khái niệm "xung đột" giữa các lần gọi.

CẬP NHẬT (sau khi rà lại với tài liệu open_clip chính thức):
- metaclip2_worldwide PHẢI dùng arch "ViT-H-14-worldwide-quickgelu" (không phải
  "ViT-H-14-worldwide") — thiếu hậu tố -quickgelu sẽ không tìm thấy pretrained
  config tương ứng, load sẽ lỗi ngay.
- encode_texts giờ luôn truyền context_length=model.context_length tường minh,
  vì SigLIP2 (64 token) và CLIP-family (77 token) không dùng chung default.
- Thay openclip_vith14_laion2b bằng DFN5B cùng kiến trúc ViT-H/14 nhưng data
  curation tốt hơn (~84% vs ~78% zero-shot ImageNet cùng compute class).
  Chưa xác nhận được tag ngắn cho DFN5B trong registry nên load qua hf-hub
  trực tiếp — đây là cách duy nhất mọi model card chính thức xác nhận hoạt động.
- siglip2_so400m: bỏ field image_size (dead code, và sai — checkpoint này chỉ
  có ở resolution ~384, không có bản 224). Resolution thật được log ra khi load
  để đối chiếu, không còn dựa vào con số gõ tay có thể sai.
- siglip2_so400m dùng được cho CẢ 3 dataset (không chỉ riêng mmimdb):
  multi-label (mmimdb) dùng sigmoid độc lập từng lớp; single-label
  (Fakeddit/CrisisMMD) dùng calibrated_logits() rồi softmax ở tầng inference,
  KHÔNG gọi sigmoid — vì sigmoid đơn điệu theo từng phần tử nên argmax không đổi.
  Quyết định sigmoid-vs-softmax là việc của inference.py theo TASK, không phải
  theo MODEL, nên wrapper chỉ cung cấp logits đã hiệu chỉnh, không tự quyết định.
"""

import torch
import open_clip


# ============================================================
# CẤU HÌNH CÁC MODEL SẼ DÙNG TRONG DỰ ÁN
# ============================================================
# Yêu cầu: pip install -U open_clip_torch  (>= 2.31.0 để có SigLIP2 / MetaCLIP2)
MODEL_CONFIGS = {
    # Dùng cho cả 3 dataset — data curation kỹ hơn là điểm mạnh của MetaCLIP 2
    # trên domain ít gặp (tweet/poster không giống ảnh web thông thường).
    "metaclip2_vith14": {
        "arch": "ViT-H-14-worldwide-quickgelu",   # FIX: thiếu "-quickgelu" sẽ crash khi load
        "pretrained": "metaclip2_worldwide",
        "family": "clip_softmax",
    },
    
    
    "dfn5b_vith14": {
        "arch": "hf-hub:apple/DFN5B-CLIP-ViT-H-14",
        "pretrained": None,          # bắt buộc None khi arch là hf-hub:...
        "tokenizer": "ViT-H-14",     # Apple dùng tokenizer CLIP chuẩn, không phải hf-hub id
        "family": "clip_softmax",
    },

    # Dùng cho cả 3 dataset. Với mmimdb (multi-label): dùng sigmoid độc lập
    # từng lớp qua calibrated_logits(). Với Fakeddit/CrisisMMD (single-label):
    # vẫn dùng calibrated_logits() nhưng softmax ở tầng inference.py.
    "siglip2_so400m": {
        "arch": "ViT-SO400M-14-SigLIP2",
        "pretrained": "webli",
        "family": "siglip_sigmoid",
    },
}


class VLMWrapper:
    """
    Bọc 1 model VLM (CLIP/SigLIP) lại thành interface đồng nhất:
    - encode_images(list[PIL.Image]) -> tensor embedding đã normalize
    - encode_texts(list[str]) -> tensor embedding đã normalize
    - similarity(image_embeds, text_embeds) -> cosine similarity thô (để debug/log)
    - calibrated_logits(image_embeds, text_embeds) -> logits đã nhân logit_scale
      + cộng logit_bias, sẵn sàng để inference.py tự chọn softmax hay sigmoid
      tuỳ theo task (single-label hay multi-label), không phải tuỳ theo model.
    Dùng chung interface này để run_experiment.py không cần biết
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
        
        # self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
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

        # Lấy calibration params thật từ model đã học, KHÔNG hardcode.
        # logit_bias chỉ tồn tại ở SigLIP-family; CLIP-family không có -> mặc định 0.
        # self.logit_scale = float(self.model.logit_scale.exp().item())
        # self.logit_bias = float(getattr(self.model, "logit_bias", torch.tensor(0.0)).item())
        self.logit_scale = float(self.model.logit_scale.exp().item())
        logit_bias = getattr(self.model, "logit_bias", None)
        self.logit_bias = float(logit_bias.item()) if logit_bias is not None else 0.0

        # Embedding dim thật, để downstream code assert thay vì crash âm thầm
        # khi cache embedding của nhiều model vào cùng một bảng cố định chiều.
        self.embed_dim = getattr(self.model.visual, "output_dim", None)

        # Resolution THẬT model dùng, đối chiếu với con số khai trong docstring/README
        # của bạn (đừng tin số gõ tay — đây chính là lỗi vừa bắt được ở siglip2).
        actual_image_size = getattr(self.model.visual, "image_size", None)

        print(f"[model_registry] -> Load xong '{model_name}': "
              f"embed_dim={self.embed_dim}, image_size={actual_image_size}, "
              f"context_length={self.model.context_length}, "
              f"logit_scale={self.logit_scale:.2f}, logit_bias={self.logit_bias:.4f}")

    @torch.no_grad()
    def encode_images(self, images: list) -> torch.Tensor:
        """images: list các PIL.Image (chưa preprocess)."""
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        embeds = self.model.encode_image(batch)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds

    @torch.no_grad()
    def encode_texts(self, texts: list) -> torch.Tensor:
        """texts: list các câu prompt (string)."""
        # FIX: truyền context_length tường minh — bắt buộc với SigLIP2 (64 token),
        # không được để tokenizer tự đoán default vì có thể không khớp model.
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
        """Giải phóng GPU memory trước khi load model tiếp theo."""
        del self.model
        
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        print(f"[model_registry] Đã giải phóng '{self.model_name}' khỏi bộ nhớ")


def load_model(model_name: str, device: str = None) -> VLMWrapper:
    """Hàm tiện ích để gọi từ run_experiment.py"""
    return VLMWrapper(model_name, device=device)


def list_available_models() -> list:
    return list(MODEL_CONFIGS.keys())


if __name__ == "__main__":
    # Smoke test: load lần lượt TỪNG model trong MODEL_CONFIGS, không chỉ 1 model
    # như bản cũ — chính cách test cũ (chỉ test siglip2) là lý do lỗi quickgelu
    # ở metaclip2 không bị phát hiện sớm.
    print("Các model khả dụng:", list_available_models())
    for name in list_available_models():
        try:
            vlm = load_model(name)
            text_embeds = vlm.encode_texts(["a photo of a cat", "a photo of a dog"])
            print(f"  [{name}] OK — text_embeds shape: {tuple(text_embeds.shape)}")
            vlm.unload()
        except Exception as e:
            print(f"  [{name}] LỖI khi load: {e}")