"""
Prompt templates cho CrisisMMD.

Humanitarian dùng đủ 8 nhãn của schema processed, kể cả missing_or_found_people.
Prompt thử nghiệm dựa vào định nghĩa lớp và tối đa 5 dòng đầu, không suy ra
phân bố toàn dataset hoặc loại lớp chỉ vì không xuất hiện trong mẫu đọc.
Mỗi lớp có cùng số prompt ngắn, được gộp thành một vector đại diện lớp.
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

# PROMPTS_INFORMATIVENESS = {
#     "informative": [
#         "a disaster-related post with text providing useful crisis information and an image showing relevant disaster conditions",

#         "a disaster-related post with text reporting destruction and an image showing damaged buildings or infrastructure",

#         "a disaster-related post with text describing dangerous conditions and an image showing visible disaster threats",

#         "a disaster-related post with text reporting damage to homes and an image showing damaged or destroyed houses",

#         "a disaster-related post with text reporting casualties or injuries and an image showing injured people or emergency medical care",

#         "a disaster-related post with text reporting missing people and an image showing a missing-person notice or a search operation",

#         "a disaster-related post with text reporting people in danger and an image showing people affected by the disaster",

#         "a disaster-related post with text reporting a wildfire and an image showing flames, smoke, or burned areas",

#         "a disaster-related post with text reporting flooding and an image showing submerged streets, buildings, or vehicles",

#         "a disaster-related post with text reporting earthquake damage and an image showing collapsed structures or debris",

#         "a disaster-related post with text reporting a landslide and an image showing displaced earth or blocked roads",

#         "a disaster-related post with text providing evacuation information and an image showing evacuation routes, notices, or people leaving affected areas",

#         "a disaster-related post with text describing rescue operations and an image showing rescuers helping affected people",

#         "a disaster-related post with text providing humanitarian aid information and an image showing shelters, relief supplies, or aid distribution",

#         "a disaster-related post with text describing the aftermath and an image showing debris, damaged neighborhoods, or displaced people",

#         "a disaster-related post with text giving specific crisis updates and an image showing a disaster map, warning bulletin, or impact statistics",
#     ],

#     "not_informative": [
#         "a post with text providing no useful crisis information and an image showing no relevant disaster information",

#         "a post with text discussing everyday life and an image showing ordinary activities unrelated to a disaster",

#         "a post with text discussing entertainment or hobbies and an image showing unrelated leisure activities",

#         "a post with text containing casual conversation and an image unrelated to disaster impacts or response",

#         "a post with text sharing personal updates unrelated to a disaster and an image showing an unrelated personal scene",

#         "a post with text expressing general emotions without specific crisis details and an image providing no disaster evidence",

#         "a post with text offering general sympathy without actionable details and an image showing only a decorative symbol or greeting",

#         "a post with text mentioning a disaster without specific information and an image unrelated to the event",

#         "a post with text discussing unrelated news and an image illustrating a different topic",

#         "a post with text advertising products or services unrelated to disaster relief and an image showing promotional material",

#         "a post with text sharing a joke unrelated to a disaster and an image showing an unrelated meme or cartoon",

#         "a post with text describing a pleasant day unrelated to a crisis and an image showing an ordinary landscape",

#         "a post with text discussing routine city life and an image showing normal streets without crisis-related context",

#         "a post with text about celebrations unrelated to a disaster and an image showing a social gathering",

#         "a post with text sharing a generic slogan without crisis details and an image showing decorative text",

#         "a post with text providing no specific disaster update and an image showing unrelated scenery or objects",
#     ],
# }

LABEL_NAMES_INFORMATIVENESS = {
    "informative": "Informative",
    "not_informative": "Not Informative",
}


# ============================================================
# TASK 2: HUMANITARIAN (8 lớp)
# ============================================================
PROMPTS_HUMANITARIAN = {
    "not_humanitarian": [
        "a casual tweet or conversation without information about disaster impacts, needs, or response",
        "a vague mention of a disaster without details about its impact or response",
        "an everyday photograph unrelated to disaster impacts or humanitarian assistance",
        "a social media image with no identifiable information about disaster needs or relief",
    ],
    "other_relevant_information": [
        "a tweet giving a general update on the location, spread, or severity of a disaster",
        "a report about an ongoing wildfire, flood, storm, or earthquake and its conditions",
        "an overview photograph showing a disaster scene or an approaching natural hazard",
        "a weather map, hazard warning, or situation update providing context about a disaster",
    ],
    "rescue_volunteering_or_donation_effort": [
        "a tweet reporting rescue operations or volunteers helping people during a disaster",
        "an appeal for donations, relief supplies, shelter, or volunteers for disaster survivors",
        "a photograph of emergency teams rescuing people from a disaster",
        "a photograph of volunteers distributing food, water, clothing, or other relief supplies",
    ],
    "infrastructure_and_utility_damage": [
        "a tweet reporting buildings, homes, roads, or bridges damaged by a disaster",
        "a report of disaster damage disrupting electricity, water, communications, or transport infrastructure",
        "a photograph of collapsed buildings, burned houses, or destroyed structures",
        "a photograph of damaged roads, broken bridges, fallen power lines, or flooded infrastructure",
    ],
    "affected_individuals": [
        "a tweet describing people displaced, evacuated, or left homeless by a disaster",
        "a report of families stranded, sheltering, or struggling with the loss of homes and belongings",
        "a photograph of disaster survivors leaving their homes or staying in temporary shelter",
        "a photograph of people affected by flooding, fire, or an earthquake and needing assistance",
    ],
    "vehicle_damage": [
        "a tweet reporting cars, trucks, buses, or boats damaged by a disaster",
        "a report of vehicles submerged, burned, overturned, or crushed during a disaster",
        "a photograph of flood-damaged cars or vehicles submerged in water",
        "a photograph of burned, crushed, or overturned vehicles after a natural disaster",
    ],
    "injured_or_dead_people": [
        "a tweet reporting people injured or killed in a disaster",
        "a report of disaster casualties, deaths, injuries, or medical treatment for victims",
        "a photograph of injured disaster victims receiving medical attention",
        "a photograph documenting human casualties or the recovery of bodies after a disaster",
    ],
    "missing_or_found_people": [
        "a tweet asking for information about a person missing after a disaster",
        "an update confirming that a missing person has been found or reunited with family",
        "a missing-person notice with a photograph and identifying details during a disaster",
        "a social media notice about locating or finding people separated during a disaster",
    ],
}

# PROMPTS_HUMANITARIAN = {
#     "not_humanitarian": [
#         "a post with casual conversation and an everyday image unrelated to disaster impacts, needs, or response",
#         "a post with a vague disaster mention and an image providing no identifiable information about disaster impacts or relief",
#     ],

#     "other_relevant_information": [
#         "a post with text updating the location, spread, or severity of a disaster and an image showing an overview of the event",
#         "a post with text reporting disaster conditions and an image showing a weather map, hazard warning, or situation update",
#     ],

#     "rescue_volunteering_or_donation_effort": [
#         "a post with text reporting rescue operations and an image showing emergency teams rescuing people during a disaster",
#         "a post with text requesting donations or volunteers and an image showing volunteers distributing relief supplies to disaster survivors",
#     ],

#     "infrastructure_and_utility_damage": [
#         "a post with text reporting disaster damage to buildings or homes and an image showing collapsed buildings, burned houses, or destroyed structures",
#         "a post with text reporting disruption to utilities or transport infrastructure and an image showing damaged roads, broken bridges, or fallen power lines",
#     ],

#     "affected_individuals": [
#         "a post with text describing displaced or evacuated people and an image showing disaster survivors leaving their homes or staying in temporary shelter",
#         "a post with text describing families stranded or losing their homes and an image showing disaster-affected people needing assistance",
#     ],

#     "vehicle_damage": [
#         "a post with text reporting disaster damage to cars, trucks, buses, or boats and an image showing damaged vehicles",
#         "a post with text describing vehicles submerged, burned, overturned, or crushed during a disaster and an image showing the affected vehicles",
#     ],

#     "injured_or_dead_people": [
#         "a post with text reporting disaster injuries and an image showing injured victims receiving medical attention",
#         "a post with text reporting disaster deaths and an image documenting human casualties or the recovery of bodies",
#     ],

#     "missing_or_found_people": [
#         "a post with text seeking a person missing after a disaster and an image showing a missing-person notice with identifying details",
#         "a post with text confirming a missing person has been found or reunited with family and an image showing a corresponding found-person or reunification notice",
#     ],
# }

LABEL_NAMES_HUMANITARIAN = {
    "not_humanitarian": "Not Humanitarian",
    "other_relevant_information": "Other Relevant Information",
    "rescue_volunteering_or_donation_effort": "Rescue/Volunteering/Donation",
    "infrastructure_and_utility_damage": "Infrastructure/Utility Damage",
    "affected_individuals": "Affected Individuals",
    "vehicle_damage": "Vehicle Damage",
    "injured_or_dead_people": "Injured or Dead People",
    "missing_or_found_people": "Missing or Found People",
}

# Cảnh báo kế thừa từ phân tích trước; chưa kiểm đếm lại dữ liệu hiện tại.
# Không dùng danh sách này để loại nhãn khỏi bộ 8 lớp.
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
