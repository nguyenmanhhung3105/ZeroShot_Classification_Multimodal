"""
Core zero-shot inference — đúng cơ chế thật của CLIP đã giải thích:
KHÔNG có "hỏi - đáp", chỉ có encode ảnh + encode text -> so cosine similarity -> chọn nhãn.
"""

import torch
from models.model_registry import VLMWrapper


def build_class_embeddings(vlm: VLMWrapper, prompt_set: dict) -> tuple:
    """
    prompt_set: dict {label: [câu prompt 1, câu prompt 2, ...]}

    Với mỗi lớp có nhiều câu prompt (ensembling), encode hết rồi lấy TRUNG BÌNH
    embedding của các câu đó làm đại diện cho lớp -> giúp kết quả ổn định hơn
    so với chỉ dùng 1 câu prompt duy nhất.

    Trả về:
        - class_embeds: tensor [n_classes, dim]
        - label_order: list label theo đúng thứ tự hàng trong class_embeds
    """
    label_order = list(prompt_set.keys())
    class_embeds_list = []

    for label in label_order:
        prompts = prompt_set[label]
        text_embeds = vlm.encode_texts(prompts)          # [n_prompts, dim]
        mean_embed = text_embeds.mean(dim=0)              # trung bình các prompt
        mean_embed = mean_embed / mean_embed.norm()        # normalize lại sau khi mean
        class_embeds_list.append(mean_embed)

    class_embeds = torch.stack(class_embeds_list)          # [n_classes, dim]
    return class_embeds, label_order


@torch.no_grad()
def predict_single_label(vlm: VLMWrapper, images: list, class_embeds: torch.Tensor,
                          label_order: list) -> list:
    """
    Dùng cho bài toán single-label (Fakeddit, CrisisMMD): argmax similarity.
    images: list PIL.Image
    Trả về: list label dự đoán (theo đúng thứ tự images đưa vào)
    """
    image_embeds = vlm.encode_images(images)                    # [n_images, dim]
    sims = vlm.similarity(image_embeds, class_embeds)            # [n_images, n_classes]
    pred_indices = sims.argmax(dim=1).cpu().tolist()
    predictions = [label_order[i] for i in pred_indices]
    return predictions, sims.cpu().numpy()


@torch.no_grad()
def predict_multi_label(vlm: VLMWrapper, images: list, class_embeds: torch.Tensor,
                         label_order: list, threshold: float = 0.22) -> list:
    """
    Dùng cho bài toán multi-label (MM-IMDb): chọn TẤT CẢ lớp có similarity vượt threshold,
    KHÔNG dùng argmax vì 1 ảnh có thể thuộc nhiều lớp cùng lúc.

    threshold: ngưỡng similarity để coi là "thuộc lớp này". Giá trị 0.22 là điểm khởi đầu
    hợp lý cho CLIP cosine similarity (thường nằm trong khoảng 0.15-0.35), CẦN tinh chỉnh
    lại dựa trên phân tích thực tế trên tập validation.
    """
    image_embeds = vlm.encode_images(images)
    sims = vlm.similarity(image_embeds, class_embeds)            # [n_images, n_classes]

    predictions = []
    for row in sims.cpu().numpy():
        pred_labels = [label_order[i] for i, score in enumerate(row) if score >= threshold]
        if not pred_labels:  # fallback: nếu không lớp nào vượt threshold, lấy lớp cao nhất
            pred_labels = [label_order[row.argmax()]]
        predictions.append(pred_labels)

    return predictions, sims.cpu().numpy()
