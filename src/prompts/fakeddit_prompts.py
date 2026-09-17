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
# PROMPTS_2WAY = {
#     0: [  # fake
#         "a fake or misleading Reddit post",
#         "a social media post spreading misinformation",
#     ],
#     1: [  # real
#         "a real and accurate Reddit post",
#         "a genuine, truthful social media post",
#     ],
# }

#------------------------------------------------------------------------------------------

PROMPTS_2WAY = {

    0: [  # fake

        # =========================
        # TEXT-FOCUSED PROMPTS
        # =========================

        "a Reddit post containing false or misleading information",
        "a Reddit title presenting fabricated or deceptive information",
        "a social media post spreading false or misleading claims",
        "a Reddit post that distorts or misrepresents a real event",
        "a Reddit title designed to create a misleading impression",
        "a satirical or parody Reddit post that should not be interpreted as factual",
        "a Reddit post containing exaggerated, deceptive, or manipulated information",
        "a Reddit post misrepresenting the original source or meaning of the content",
        "a Reddit title that does not accurately describe the associated content",
        "a social media caption creating a false connection with the associated image",
        "a misleading caption that gives the image a false or incorrect meaning",
        "a Reddit post presenting content in a false or deceptive context",

        # =========================
        # IMAGE-FOCUSED PROMPTS
        # =========================

        "a photo containing manipulated, altered, or fabricated visual content",
        "a photo digitally edited to create a false or misleading impression",
        "an image containing visually manipulated or deceptive content",
        "a manipulated image presented as if it were authentic",
        "a photo used in a false or misleading context",
        "an image used to support a misleading social media post",
        "a photo that does not accurately support the accompanying Reddit title",
        "an image whose meaning is misrepresented by the accompanying caption",
        "a meme, parody, or altered image presented as factual content",
        "a fabricated or edited image intended to mislead the viewer",
        "an image creating a false impression about an event, object, or person",
        "a misleading social media image associated with false or deceptive information",
    ],

    1: [  # real

        # =========================
        # TEXT-FOCUSED PROMPTS
        # =========================

        "a Reddit post containing accurate and truthful information",
        "a Reddit title accurately describing a real event, object, or scene",
        "a factual social media post reporting genuine information",
        "a Reddit post presenting information without misleading manipulation",
        "a truthful Reddit title that accurately represents the original content",
        "a social media post describing a real event or situation accurately",
        "a Reddit post presenting authentic and correctly contextualized information",
        "a factual post whose title accurately reflects the meaning of the content",
        "a Reddit title that correctly describes the associated image",
        "a social media caption that accurately matches the associated image",
        "a truthful caption providing the correct context for an image",
        "an unusual or surprising but genuine and factual Reddit post",

        # =========================
        # IMAGE-FOCUSED PROMPTS
        # =========================

        "a real and authentic photo showing an actual event, object, or scene",
        "a genuine photo accurately representing what really happened",
        "an authentic image that has not been misleadingly manipulated",
        "a real photo presented in its correct context",
        "a photo that accurately supports the accompanying Reddit title",
        "an image whose content correctly matches the accompanying caption",
        "a genuine image used to support accurate social media information",
        "an authentic photo without deceptive visual manipulation",
        "a real image accurately documenting an event, object, or situation",
        "a factual social media image presented with the correct meaning",
        "an unusual or surprising but genuine photograph",
        "an authentic image associated with truthful and correctly contextualized information",
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
