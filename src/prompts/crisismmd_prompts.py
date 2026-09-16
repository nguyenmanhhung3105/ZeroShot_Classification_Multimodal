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
# PROMPTS_INFORMATIVENESS = {
#     "informative": [
#         "a tweet with useful, informative content about a disaster",
#         "a photo showing relevant information about a crisis event",
#     ],
#     "not_informative": [
#         "a tweet with no useful information about the disaster",
#         "an irrelevant photo not related to disaster response",
#     ],
# }

PROMPTS_INFORMATIVENESS = {
    "informative": [
        "a tweet with useful, informative content about a disaster",
        "a tweet reporting destruction or damage caused by a disaster",
        "a tweet describing a dangerous or devastating crisis situation",
        "a tweet describing fear, panic, suffering, or distress during a disaster",
        "a tweet reporting collapsed buildings, damaged homes, or destroyed infrastructure",
        "a tweet reporting casualties, injuries, missing people, or people in danger",
        "a tweet describing severe damage caused by fire, flood, earthquake, tsunami, or landslide",
        "a tweet reporting emergency conditions during a natural disaster",
        "a tweet providing information about evacuation, rescue, or emergency response",
        "a tweet describing the aftermath and destruction caused by a disaster",
        "a tweet containing important updates about an ongoing crisis",
        "a tweet providing useful humanitarian information about a disaster",

        "a photo showing relevant information about a crisis event",
        "a photo showing useful visual information about a disaster",
        "a photo showing destruction caused by a natural disaster",
        "a photo showing buildings or houses damaged or destroyed by a disaster",
        "a photo showing collapsed buildings and damaged infrastructure",
        "a photo showing fire, wildfire, or severe burning during a disaster",
        "a photo showing flooding, tsunami, or dangerous flood water",
        "a photo showing a landslide, earthquake damage, or collapsed terrain",
        "a photo showing a devastated city or severely damaged neighborhood",
        "a photo showing ruins, debris, destruction, or disaster aftermath",
        "a photo showing people affected by a dangerous disaster situation",
        "a photo showing rescue workers, emergency response, or disaster relief",
        "a photo containing visible information about an ongoing natural disaster",
        "a photo showing dangerous conditions during a crisis",
        "a photo documenting the aftermath of a major disaster",
    ],

    "not_informative": [
        "a tweet with no useful information about the disaster",
        "a tweet unrelated to disasters, emergencies, or dangerous events",
        "a casual tweet about everyday life with no crisis information",
        "a tweet expressing happiness, love, joy, or positive emotions unrelated to a disaster",
        "a tweet describing peaceful or pleasant situations with no danger",
        "a tweet containing ordinary conversation unrelated to a crisis",
        "a tweet about entertainment, hobbies, or personal activities unrelated to a disaster",
        "a tweet containing positive and cheerful content with no emergency information",
        "a tweet discussing unrelated news with no information about a disaster",
        "a tweet with no information about destruction, danger, rescue, or emergency response",
        "an irrelevant tweet that provides no useful crisis-related information",

        "an irrelevant photo not related to disaster response",
        "a photo unrelated to disasters, emergencies, or crisis events",
        "a normal everyday photo with no disaster-related information",
        "a peaceful and beautiful outdoor scene with no signs of danger",
        "a photo showing a clear blue sky and pleasant weather",
        "a photo showing healthy trees, nature, or a calm landscape",
        "a bright and pleasant scene with no destruction or emergency",
        "a photo showing normal buildings and streets with no visible disaster damage",
        "a photo showing ordinary daily life unrelated to a crisis",
        "a happy or cheerful photo unrelated to a disaster",
        "a photo containing news or text unrelated to a disaster",
        "an irrelevant image that provides no useful information about a crisis",
        "a photo with no visible destruction, danger, emergency, or disaster response",
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
