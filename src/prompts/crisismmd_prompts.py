"""
Prompt templates cho CrisisMMD.

LƯU Ý QUAN TRỌNG (đã phân tích trước đó):
- Task Humanitarian có phân bố CỰC LỆCH sau khi lọc mismatch + confidence threshold.
- Lớp "missing_or_found_people" có 0 mẫu sau lọc -> KHÔNG đưa vào prompt set,
  vì đưa vào cũng không có ground-truth để đánh giá, chỉ gây nhiễu (model có thể
  dự đoán ra lớp này cho các mẫu thuộc lớp khác, không đo được recall của nó).
- Các lớp affected_individuals (8), vehicle_damage (7), injured_or_dead_people (6)
  có cỡ mẫu quá nhỏ -> vẫn giữ trong prompt set để model có thể dự đoán,
  nhưng khi báo cáo kết quả cần ghi rõ cỡ mẫu quá nhỏ để kết luận đáng tin cậy.
"""

# ============================================================
# TASK 1: INFORMATIVENESS (binary)
# ============================================================
PROMPTS_INFORMATIVENESS = {
    "informative": [
        "a tweet with useful, informative content about a disaster",
        "a photo showing relevant information about a crisis event",
    ],
    "not_informative": [
        "a tweet with no useful information about the disaster",
        "an irrelevant photo not related to disaster response",
    ],
}

LABEL_NAMES_INFORMATIVENESS = {
    "informative": "Informative",
    "not_informative": "Not Informative",
}


# ============================================================
# TASK 2: HUMANITARIAN (7 lớp — đã loại missing_or_found_people vì 0 mẫu)
# ============================================================
PROMPTS_HUMANITARIAN = {
    "not_humanitarian": [
        "a tweet not related to humanitarian aid or disaster relief",
        "content irrelevant to disaster response efforts",
    ],
    "other_relevant_information": [
        "general information related to the disaster event",
        "a tweet providing context or updates about the crisis",
    ],
    "rescue_volunteering_or_donation_effort": [
        "a photo of rescue teams helping disaster victims",
        "a tweet about volunteering or donation efforts for disaster relief",
    ],
    "infrastructure_and_utility_damage": [
        "a photo showing damaged buildings, roads, or public infrastructure",
        "a tweet reporting damage to utilities or infrastructure from a disaster",
    ],
    "affected_individuals": [
        "a photo of people directly affected by the disaster, such as evacuees",
        "a tweet describing individuals impacted by the crisis",
    ],
    "vehicle_damage": [
        "a photo of a car or vehicle damaged by the disaster",
        "a tweet reporting vehicle damage from the crisis event",
    ],
    "injured_or_dead_people": [
        "a photo or report of people injured or killed by the disaster",
        "a tweet reporting casualties from the crisis event",
    ],
}

LABEL_NAMES_HUMANITARIAN = {
    "not_humanitarian": "Not Humanitarian",
    "other_relevant_information": "Other Relevant Information",
    "rescue_volunteering_or_donation_effort": "Rescue/Volunteering/Donation",
    "infrastructure_and_utility_damage": "Infrastructure/Utility Damage",
    "affected_individuals": "Affected Individuals",
    "vehicle_damage": "Vehicle Damage",
    "injured_or_dead_people": "Injured or Dead People",
}

# Các lớp có cỡ mẫu quá nhỏ (<30 mẫu theo phân tích trước) — dùng để cảnh báo
# khi in kết quả đánh giá, KHÔNG dùng để loại khỏi prompt set
LOW_SAMPLE_WARNING_CLASSES = {
    "affected_individuals",
    "vehicle_damage",
    "injured_or_dead_people",
}


def get_prompt_set(task: str = "informativeness") -> dict:
    if task == "informativeness":
        return PROMPTS_INFORMATIVENESS
    elif task == "humanitarian":
        return PROMPTS_HUMANITARIAN
    else:
        raise ValueError(f"task phải là 'informativeness' hoặc 'humanitarian', nhận được: {task}")


def get_label_names(task: str = "informativeness") -> dict:
    if task == "informativeness":
        return LABEL_NAMES_INFORMATIVENESS
    elif task == "humanitarian":
        return LABEL_NAMES_HUMANITARIAN
    else:
        raise ValueError(f"task phải là 'informativeness' hoặc 'humanitarian', nhận được: {task}")
