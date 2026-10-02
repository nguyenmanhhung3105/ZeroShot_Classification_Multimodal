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
#         "a tweet with useful, informative content about a disaster and a photo showing relevant information about a crisis event",
#         "a tweet reporting destruction or damage caused by a disaster and a photo showing destruction caused by a natural disaster",
#         "a tweet describing a dangerous or devastating crisis situation and a photo showing dangerous conditions during a crisis",
#         "a tweet describing fear, panic, suffering, or distress during a disaster and a photo showing people affected by a dangerous disaster situation",
#         "a tweet reporting collapsed buildings, damaged homes, or destroyed infrastructure and a photo showing buildings or houses damaged or destroyed by a disaster",
#         "a tweet reporting casualties, injuries, missing people, or people in danger and a photo showing people affected by a dangerous disaster situation",
#         "a tweet describing severe damage caused by fire, flood, earthquake, tsunami, or landslide and a photo showing fire, flood, earthquake damage, tsunami, or landslide destruction",
#         "a tweet reporting emergency conditions during a natural disaster and a photo containing visible information about an ongoing natural disaster",
#         "a tweet providing information about evacuation, rescue, or emergency response and a photo showing rescue workers, emergency response, or disaster relief",
#         "a tweet describing the aftermath and destruction caused by a disaster and a photo showing ruins, debris, destruction, or disaster aftermath",
#         "a tweet containing important updates about an ongoing crisis and a photo documenting the aftermath of a major disaster",
#         "a tweet providing useful humanitarian information about a disaster and a photo showing useful visual information about a disaster",
#         "a tweet describing traces or visible signs left by a disaster and a photo showing collapsed buildings and damaged infrastructure",
#         "a tweet reporting the consequences caused by a disaster and a photo showing a devastated city or severely damaged neighborhood",
#         "a tweet mentioning destruction left by a disaster and a photo showing the aftermath of a natural disaster",
#         "a tweet reporting disaster statistics, updates, or important crisis information and a photo showing a disaster-related news report, bulletin, chart, or infographic",
#     ],

#     "not_informative": [
#         "a tweet with no useful information about the disaster and an irrelevant photo not related to disaster response",
#         "a tweet unrelated to disasters, emergencies, or dangerous events and a photo unrelated to disasters, emergencies, or crisis events",
#         "a casual tweet about everyday life with no crisis information and a normal everyday photo with no disaster-related information",
#         "a tweet expressing happiness, love, joy, or positive emotions unrelated to a disaster and a happy or cheerful photo unrelated to a disaster",
#         "a tweet describing peaceful or pleasant situations with no danger and a peaceful and beautiful outdoor scene with no signs of danger",
#         "a tweet containing ordinary conversation unrelated to a crisis and a photo showing ordinary daily life unrelated to a crisis",
#         "a tweet about entertainment, hobbies, or personal activities unrelated to a disaster and a photo unrelated to disaster or emergency events",
#         "a tweet containing positive and cheerful content with no emergency information and a bright and pleasant scene with no destruction or emergency",
#         "a tweet discussing unrelated news with no information about a disaster and a photo containing news or text unrelated to a disaster",
#         "a tweet with no information about destruction, danger, rescue, or emergency response and a photo with no visible destruction, danger, emergency, or disaster response",
#         "an irrelevant tweet that provides no useful crisis-related information and an irrelevant image that provides no useful information about a crisis",
#         "a tweet about peaceful weather or a pleasant day and a photo showing a clear blue sky and pleasant weather",
#         "a tweet about beauty, nature, or calm daily life and a photo showing healthy trees, nature, or a calm landscape",
#         "a tweet unrelated to disaster damage or humanitarian response and a photo showing normal buildings and streets with no visible disaster damage",
#     ],
# }


LABEL_NAMES_INFORMATIVENESS = {
    "informative": "Informative",
    "not_informative": "Not Informative",
}


# ============================================================
# TASK 2: HUMANITARIAN (8 lớp)
# ============================================================
# PROMPTS_HUMANITARIAN = {
#     "not_humanitarian": [
#         "a casual tweet or conversation without information about disaster impacts, needs, or response",
#         "a vague mention of a disaster without details about its impact or response",
#         "an everyday photograph unrelated to disaster impacts or humanitarian assistance",
#         "a social media image with no identifiable information about disaster needs or relief",
#     ],
#     "other_relevant_information": [
#         "a tweet giving a general update on the location, spread, or severity of a disaster",
#         "a report about an ongoing wildfire, flood, storm, or earthquake and its conditions",
#         "an overview photograph showing a disaster scene or an approaching natural hazard",
#         "a weather map, hazard warning, or situation update providing context about a disaster",
#     ],
#     "rescue_volunteering_or_donation_effort": [
#         "a tweet reporting rescue operations or volunteers helping people during a disaster",
#         "an appeal for donations, relief supplies, shelter, or volunteers for disaster survivors",
#         "a photograph of emergency teams rescuing people from a disaster",
#         "a photograph of volunteers distributing food, water, clothing, or other relief supplies",
#     ],
#     "infrastructure_and_utility_damage": [
#         "a tweet reporting buildings, homes, roads, or bridges damaged by a disaster",
#         "a report of disaster damage disrupting electricity, water, communications, or transport infrastructure",
#         "a photograph of collapsed buildings, burned houses, or destroyed structures",
#         "a photograph of damaged roads, broken bridges, fallen power lines, or flooded infrastructure",
#     ],
#     "affected_individuals": [
#         "a tweet describing people displaced, evacuated, or left homeless by a disaster",
#         "a report of families stranded, sheltering, or struggling with the loss of homes and belongings",
#         "a photograph of disaster survivors leaving their homes or staying in temporary shelter",
#         "a photograph of people affected by flooding, fire, or an earthquake and needing assistance",
#     ],
#     "vehicle_damage": [
#         "a tweet reporting cars, trucks, buses, or boats damaged by a disaster",
#         "a report of vehicles submerged, burned, overturned, or crushed during a disaster",
#         "a photograph of flood-damaged cars or vehicles submerged in water",
#         "a photograph of burned, crushed, or overturned vehicles after a natural disaster",
#     ],
#     "injured_or_dead_people": [
#         "a tweet reporting people injured or killed in a disaster",
#         "a report of disaster casualties, deaths, injuries, or medical treatment for victims",
#         "a photograph of injured disaster victims receiving medical attention",
#         "a photograph documenting human casualties or the recovery of bodies after a disaster",
#     ],
#     "missing_or_found_people": [
#         "a tweet asking for information about a person missing after a disaster",
#         "an update confirming that a missing person has been found or reunited with family",
#         "a missing-person notice with a photograph and identifying details during a disaster",
#         "a social media notice about locating or finding people separated during a disaster",
#     ],
# }

PROMPTS_HUMANITARIAN = {
    "not_humanitarian": [
        # tweet
        "a casual tweet or everyday conversation that does not mention any damage, victims, or relief efforts from a disaster",
        "a tweet expressing a personal opinion, a prayer, or a joke about a hurricane or disaster without sharing any useful humanitarian information",
        "a tweet about politics, celebrities, or general news commentary that has no connection to disaster impacts or assistance",
        "a promotional or advertising tweet that happens to use disaster-related hashtags but offers no real information",
        "a vague tweet that only mentions the word hurricane, earthquake, or wildfire without describing what is happening",
        "a tweet about sports, entertainment, or daily life that is unrelated to people affected by a disaster",
        # photo
        "an ordinary photograph of daily life, such as people, food, pets, or streets, with no sign of any disaster",
        "a meme, cartoon, or illustrated graphic that contains no real information about disaster impacts or relief",
        "a screenshot of plain text, a logo, or a banner that provides no visible information about damage or humanitarian needs",
        "a photograph of a politician, celebrity, or speaker at a press event rather than a disaster scene",
        "a calm and peaceful outdoor scene with a clear sky and no visible damage, danger, or emergency",
        "an advertisement or product image that has no relation to disaster impacts or assistance",
    ],

    "other_relevant_information": [
        # tweet
        "a tweet giving a general update about where a hurricane, wildfire, flood, or earthquake is happening and how serious it is",
        "a tweet sharing a weather forecast, storm track, or official warning that helps people understand the situation",
        "a tweet reporting the current conditions of an ongoing disaster, such as rising water, spreading fire, or strong winds",
        "a tweet from a news outlet or authority summarizing the overall situation of a disaster without focusing on victims or damage",
        "a tweet describing an earthquake or storm in general terms, such as its magnitude, location, or expected path",
        "a tweet passing along useful situational information about a disaster that does not fit a specific humanitarian category",
        # photo
        "a wide overview photograph showing a disaster scene such as a wildfire, flooded area, or damaged landscape from a distance",
        "a satellite image or aerial view showing the extent of a hurricane, flood, or fire",
        "a weather map or radar image showing the path and intensity of an approaching storm",
        "a screenshot of an official hazard warning, evacuation map, or situation report about a disaster",
        "a photograph of dark storm clouds, rising floodwater, or smoke that shows a disaster developing without focusing on people or buildings",
        "an informational graphic or chart that provides context about an ongoing natural disaster",
    ],

    "rescue_volunteering_or_donation_effort": [
        # tweet
        "a tweet reporting that rescue teams or emergency crews are saving people from floodwater, fire, or collapsed buildings",
        "a tweet showing gratitude to volunteers, firefighters, soldiers, or neighbors who are helping disaster survivors",
        "a tweet asking people to donate money, food, water, clothing, or supplies to help disaster victims",
        "a tweet calling for volunteers or announcing a fundraising campaign for communities recovering from a disaster",
        "a tweet announcing that a shelter, relief center, or aid distribution point is open for people in need",
        "a tweet describing charities, the Red Cross, or local organizations delivering help to affected areas",
        # photo
        "a photograph of rescue workers or boats helping people escape from flooded streets",
        "a photograph of firefighters or emergency responders carrying out a rescue during a disaster",
        "a photograph of volunteers handing out food, bottled water, blankets, or clothing to survivors",
        "a photograph of people loading, sorting, or delivering relief supplies and donations",
        "a photograph of volunteers cleaning up debris or helping rebuild homes after a disaster",
        "a poster or flyer that asks for donations, volunteers, or support for disaster relief",
    ],

    "infrastructure_and_utility_damage": [
        # tweet
        "a tweet reporting that houses, buildings, or schools were damaged or destroyed by a disaster",
        "a tweet describing roads, bridges, or highways that are flooded, cracked, or blocked after a disaster",
        "a tweet reporting power outages, broken water supply, or lost phone and internet service caused by a disaster",
        "a tweet describing collapsed walls, torn roofs, or fallen trees and power lines after a storm",
        "a tweet reporting that airports, railways, or public transportation were shut down because of damage",
        "a tweet describing entire neighborhoods or towns left in ruins after an earthquake, fire, or hurricane",
        # photo
        "a photograph of collapsed buildings or houses reduced to rubble after an earthquake or hurricane",
        "a photograph of homes burned to the ground by a wildfire, leaving only ashes and ruins",
        "a photograph of a broken bridge, cracked road, or highway washed away by flooding",
        "a photograph of fallen power lines, broken utility poles, or electrical equipment damaged by a storm",
        "a photograph of a street or building heavily flooded, with water filling roads and entering structures",
        "a photograph of a roof torn off, walls destroyed, or a public building severely damaged by a disaster",
    ],

    "affected_individuals": [
        # tweet
        "a tweet describing families who lost their homes and are now displaced or living without shelter after a disaster",
        "a tweet about people who were evacuated and are staying in shelters or with relatives",
        "a tweet describing residents who are stranded, hungry, or waiting for help because of a flood or storm",
        "a tweet sharing the emotional struggle of survivors who lost their belongings, livelihoods, or loved places",
        "a tweet reporting how many people were affected, displaced, or left without basic necessities by a disaster",
        "a tweet describing survivors trying to cope and rebuild their lives after a disaster",
        # photo
        "a photograph of families carrying their belongings as they leave their homes during an evacuation",
        "a photograph of survivors sitting or sleeping in a crowded temporary shelter or relief camp",
        "a photograph of residents standing in knee-deep floodwater in front of their damaged homes",
        "a photograph of people looking through the remains of their destroyed house for belongings",
        "a photograph of distressed or exhausted disaster survivors waiting for assistance",
        "a photograph of children and elderly people affected by a disaster and in need of help",
    ],

    "vehicle_damage": [
        # tweet
        "a tweet reporting that cars, trucks, or buses were damaged by floodwater, fire, or falling debris",
        "a tweet describing vehicles that were swept away, submerged, or buried during a disaster",
        "a tweet about boats or ships damaged, washed ashore, or destroyed by a hurricane or storm",
        "a tweet reporting cars crushed by fallen trees, collapsed walls, or broken buildings",
        "a tweet showing concern for vehicles left abandoned, flooded, or burned during a disaster",
        "a tweet reporting that many vehicles were destroyed in a wildfire, earthquake, or flood",
        # photo
        "a close-up photograph of a car partly or fully submerged in floodwater",
        "a photograph of rows of flooded cars standing in a water-filled street",
        "a photograph of burned-out vehicles left behind after a wildfire",
        "a photograph of a car crushed by a fallen tree, debris, or collapsed structure",
        "a photograph of overturned trucks or buses damaged by a powerful storm or earthquake",
        "a photograph of boats thrown onto land or wrecked by a hurricane",
    ],

    "injured_or_dead_people": [
        # tweet
        "a tweet reporting that people were killed in a disaster and mentioning the number of deaths",
        "a tweet reporting that many people were injured and are being treated at hospitals",
        "a tweet sharing the growing death toll or casualty figures from an earthquake, hurricane, or flood",
        "a tweet mourning the victims who lost their lives in a disaster",
        "a tweet describing medical teams treating wounded survivors after a disaster",
        "a tweet reporting that bodies were recovered or victims were found dead after a disaster",
        # photo
        "a photograph of injured survivors being treated by doctors, nurses, or paramedics after a disaster",
        "a photograph of a wounded person being carried on a stretcher or helped into an ambulance",
        "a photograph of victims lying injured or lifeless among the debris of a disaster",
        "a photograph of rescuers recovering bodies from rubble, floodwater, or burned areas",
        "a photograph of people grieving or mourning beside victims who died in a disaster",
        "a photograph of a field hospital or medical tent caring for injured disaster victims",
    ],

    "missing_or_found_people": [
        # tweet
        "a tweet asking for help finding a family member or friend who has gone missing after a disaster",
        "a tweet sharing the name, age, and last known location of a person missing after a disaster",
        "a tweet announcing that a missing person has been found safe and reunited with their family",
        "a tweet asking people to share information about loved ones who they cannot contact after a disaster",
        "a tweet reporting that rescuers are still searching for people missing under rubble or floodwater",
        "a tweet confirming the safety of people who were previously reported missing",
        # photo
        "a missing person poster showing a photograph, name, and contact details",
        "a social media post with a person's photo asking whether anyone has seen them after a disaster",
        "a photograph of search teams looking for missing people among ruins or flooded areas",
        "a photograph of family members happily reunited after being separated during a disaster",
        "a list or board of names of people who are missing or have been found safe",
        "a photograph of someone holding a picture of a missing relative while searching for information",
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
