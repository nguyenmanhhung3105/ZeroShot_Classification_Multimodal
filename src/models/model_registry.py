"""
Model Registry — nơi khai báo và load các Vision-Language Model dùng cho zero-shot.

NGUYÊN TẮC QUAN TRỌNG (đã thống nhất trong quá trình thiết kế dự án):
- Mỗi model chỉ được load ĐÚNG 1 LẦN, tái sử dụng xuyên suốt nhiều dataset và nhiều prompt set.
- Đây là inference thuần túy (forward pass), không có bước train/fine-tune nào —
  vì vậy không có khái niệm "xung đột" giữa các lần gọi.
"""

import torch
import open_clip


# ============================================================
# CẤU HÌNH CÁC MODEL SẼ DÙNG TRONG DỰ ÁN
# ============================================================
MODEL_CONFIGS = {
    "clip_vitb32": {
        "arch": "ViT-B-32",
        "pretrained": "openai",
        "image_size": 224,
    },
    "clip_vitl14": {
        "arch": "ViT-L-14",
        "pretrained": "openai",
        "image_size": 224,
    },
    "siglip_vitb16": {
        "arch": "ViT-B-16-SigLIP",
        "pretrained": "webli",
        "image_size": 224,
    },
}


class VLMWrapper:
    """
    Bọc 1 model VLM (CLIP/SigLIP) lại thành interface đồng nhất:
    - encode_images(list[PIL.Image]) -> tensor embedding đã normalize
    - encode_texts(list[str]) -> tensor embedding đã normalize
    - similarity(image_embeds, text_embeds) -> ma trận cosine similarity

    Dùng chung interface này để run_experiment.py không cần biết
    bên trong là CLIP hay SigLIP, tránh việc phải viết code riêng cho từng model.
    """

    def __init__(self, model_name: str, device: str = None):
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Model '{model_name}' chưa được khai báo trong MODEL_CONFIGS. "
                f"Các model hợp lệ: {list(MODEL_CONFIGS.keys())}"
            )

        config = MODEL_CONFIGS[model_name]
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[model_registry] Đang load model '{model_name}' ({config['arch']}, "
              f"pretrained={config['pretrained']}) lên {self.device} ...")

        # force_quick_gelu=True: khớp đúng kiến trúc gốc mà OpenAI dùng khi train CLIP,
        # tránh cảnh báo "QuickGELU mismatch" và tránh lệch nhẹ kết quả zero-shot.
        # Chỉ cần cho các checkpoint pretrained="openai"; SigLIP không dùng QuickGELU nên bỏ qua.
        extra_kwargs = {"force_quick_gelu": True} if config["pretrained"] == "openai" else {}

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config["arch"], pretrained=config["pretrained"], **extra_kwargs
        )
        self.tokenizer = open_clip.get_tokenizer(config["arch"])

        self.model.to(self.device)
        self.model.eval()  # QUAN TRỌNG: chế độ eval, tắt dropout/batchnorm update

        print(f"[model_registry] - Load xong '{model_name}'")

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
        tokens = self.tokenizer(texts).to(self.device)
        embeds = self.model.encode_text(tokens)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds

    @staticmethod
    def similarity(image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """Trả về ma trận [n_images, n_texts] điểm cosine similarity."""
        return image_embeds @ text_embeds.T

    def unload(self):
        """Giải phóng GPU memory trước khi load model tiếp theo."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[model_registry] Đã giải phóng '{self.model_name}' khỏi bộ nhớ")


def load_model(model_name: str, device: str = None) -> VLMWrapper:
    """Hàm tiện ích để gọi từ run_experiment.py"""
    return VLMWrapper(model_name, device=device)


def list_available_models() -> list:
    return list(MODEL_CONFIGS.keys())


if __name__ == "__main__":
    # Test nhanh: load thử 1 model, encode 1 câu text, in shape ra kiểm tra
    print("Các model khả dụng:", list_available_models())
    vlm = load_model("siglip_vitb16")
    text_embeds = vlm.encode_texts(["a photo of a cat", "a photo of a dog"])
    print("Shape text embedding:", text_embeds.shape)
    vlm.unload()
