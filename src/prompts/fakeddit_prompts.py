"""
Prompt templates cho Fakeddit — bài toán phát hiện tin giả trên Reddit.

Mỗi lớp có 1 danh sách câu prompt (không chỉ 1 câu duy nhất) để hỗ trợ
"prompt ensembling" — khi encode, có thể lấy trung bình embedding của
nhiều câu prompt cho cùng 1 lớp, giúp kết quả ổn định hơn so với chỉ
dùng 1 câu template cứng.
"""

# ============================================================
# BỘ PROMPT CHO 2_WAY_LABEL (binary: real vs fake)
# ============================================================
PROMPTS_2WAY = {
    0: [  # fake
        "a fake or misleading Reddit post",
        "a social media post spreading misinformation",
    ],
    1: [  # real
        "a real and accurate Reddit post",
        "a genuine, truthful social media post",
    ],
}

LABEL_NAMES_2WAY = {0: "fake", 1: "real"}


# ============================================================
# BỘ PROMPT CHO 6_WAY_LABEL (chi tiết loại giả mạo)
# ============================================================
PROMPTS_6WAY = {
    0: [  # True
        "a real news post that is completely true and accurate",
        "an authentic photo with an accurate, truthful caption",
    ],
    1: [  # Satire/Parody
        "a satirical or parody post meant to be humorous, not taken seriously",
        "a joke post that mimics real news for comedic effect",
    ],
    2: [  # False Connection
        "a post where the image and caption are unrelated to each other",
        "a real photo paired with a caption that describes something different",
    ],
    3: [  # Imposter Content
        "a post impersonating a legitimate news source or organization",
        "fake content pretending to come from a trustworthy source",
    ],
    4: [  # Manipulated Content
        "a photo that has been digitally edited or manipulated",
        "an image altered with photo editing to deceive viewers",
    ],
    5: [  # Misleading Content
        "a post that misrepresents facts in a misleading way",
        "genuine information presented out of context to mislead readers",
    ],
}

LABEL_NAMES_6WAY = {
    0: "True",
    1: "Satire/Parody",
    2: "False Connection",
    3: "Imposter Content",
    4: "Manipulated Content",
    5: "Misleading Content",
}


def get_prompt_set(task: str = "6way") -> dict:
    """
    task: "2way" hoặc "6way"
    Trả về dict {label_id: [danh sách câu prompt]}
    """
    if task == "2way":
        return PROMPTS_2WAY
    elif task == "6way":
        return PROMPTS_6WAY
    else:
        raise ValueError(f"task phải là '2way' hoặc '6way', nhận được: {task}")


def get_label_names(task: str = "6way") -> dict:
    if task == "2way":
        return LABEL_NAMES_2WAY
    elif task == "6way":
        return LABEL_NAMES_6WAY
    else:
        raise ValueError(f"task phải là '2way' hoặc '6way', nhận được: {task}")
