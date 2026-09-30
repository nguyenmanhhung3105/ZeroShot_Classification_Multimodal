"""
Prompt templates cho MM-IMDb — phân loại thể loại phim (multi-label, 23 lớp).

LƯU Ý: đây là bài toán multi-label, KHÔNG dùng argmax như 2 dataset kia.
Mỗi genre có prompt riêng, khi encode xong sẽ dùng THRESHOLD (không phải argmax)
để quyết định 1 phim thuộc những genre nào — 1 phim có thể có 0, 1, hoặc nhiều genre.
"""

# Cùng 23 nhãn với VALID_GENRES trong preprocessing: giữ Short, không có News.
GENRES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History",
    "Horror", "Music", "Musical", "Mystery", "Short", "Romance",
    "Sci-Fi", "Sport", "Thriller", "War", "Western",
]

# Bản thử nghiệm: mỗi genre có 4 câu ngắn về thể loại, plot và poster.
# Không chép tên phim, nhân vật hoặc nội dung cụ thể của các mẫu đánh giá.
# Các genre độc lập và có thể cùng xuất hiện; Short là thời lượng, Animation
# là hình thức thể hiện, không ép chúng vào một kiểu cốt truyện duy nhất.
PROMPTS_GENRES = {
    "Action": [
        "a movie poster for an action film",
        "an action movie featuring physical conflict, chases, or daring confrontations",
        "a film about characters fighting opponents and surviving dangerous encounters",
        "an action film with combat, martial arts, stunts, or high-speed pursuits",
    ],
    "Adventure": [
        "a movie poster for an adventure film",
        "an adventure movie about a journey, quest, or exploration of unfamiliar places",
        "a film about characters overcoming obstacles while pursuing a difficult goal",
        "an adventure story involving travel, discovery, danger, and unexpected encounters",
    ],
    "Animation": [
        "a movie poster for an animated film",
        "an animated movie told through drawn, computer-generated, or stop-motion characters",
        "a film presenting its story through animation rather than live-action performances",
        "an animated feature or short film with stylized characters and settings",
    ],
    "Biography": [
        "a movie poster for a biographical film",
        "a biographical movie portraying the life of a real person",
        "a film about a real individual's personal struggles, relationships, and achievements",
        "a biopic dramatizing important episodes in an actual person's life",
    ],
    "Comedy": [
        "a movie poster for a comedy film",
        "a comedy movie built around humorous situations and amusing characters",
        "a film about misunderstandings, awkward encounters, or absurd situations played for laughs",
        "a comic story using witty dialogue, satire, slapstick, or everyday humor",
    ],
    "Crime": [
        "a movie poster for a crime film",
        "a crime movie about criminals, detectives, or people drawn into illegal activities",
        "a film about a robbery, murder, criminal organization, or police investigation",
        "a story centered on committing crimes, pursuing offenders, or facing criminal consequences",
    ],
    "Documentary": [
        "a movie poster for a documentary film",
        "a documentary examining real people, events, or issues",
        "a nonfiction film presenting interviews, observations, or archival evidence",
        "a documentary recording real experiences or investigating a factual subject",
    ],
    "Drama": [
        "a movie poster for a drama film",
        "a drama exploring personal conflicts, difficult choices, and human relationships",
        "a film about characters facing emotional struggles and significant changes in their lives",
        "a character-driven dramatic story about family, society, loss, or personal growth",
    ],
    "Family": [
        "a movie poster for a family film",
        "a family movie intended to be enjoyed by children and adults together",
        "a family-oriented story about friendship, belonging, courage, or growing up",
        "a film for a broad family audience with accessible characters and themes",
    ],
    "Fantasy": [
        "a movie poster for a fantasy film",
        "a fantasy movie involving magic, mythical beings, or supernatural powers",
        "a film about characters entering an enchanted world or encountering magical forces",
        "a fantasy story where spells, legends, or impossible creatures shape events",
    ],
    "Film-Noir": [
        "a movie poster for a film-noir crime drama",
        "a film-noir story involving deception, moral ambiguity, and a doomed protagonist",
        "a dark crime film about betrayal, corruption, or a dangerous entanglement",
        "a noir film with shadowy imagery, cynical characters, and a fatalistic atmosphere",
    ],
    "History": [
        "a movie poster for a historical film",
        "a historical movie portraying significant events or social conditions of an earlier era",
        "a film dramatizing actual historical events and the people involved",
        "a story grounded in documented political, cultural, or social history",
    ],
    "Horror": [
        "a movie poster for a horror film",
        "a horror movie intended to evoke fear, dread, or terror",
        "a film about characters facing a haunting, monstrous threat, or terrifying violence",
        "a horror story with an ominous atmosphere and frightening encounters",
    ],
    "Music": [
        "a movie poster for a film about music or musicians",
        "a music-centered movie about performers, bands, composers, or musical careers",
        "a film about creating music, rehearsing, recording, or performing in concerts",
        "a story in which musical performance and the lives of musicians are central",
    ],
    "Musical": [
        "a movie poster for a musical film",
        "a musical in which characters sing or dance as part of telling the story",
        "a film whose songs express characters' feelings and advance the narrative",
        "a musical movie combining dramatic scenes with staged song-and-dance numbers",
    ],
    "Mystery": [
        "a movie poster for a mystery film",
        "a mystery movie about uncovering a hidden truth or explaining puzzling events",
        "a film in which characters follow clues to solve a disappearance, secret, or unexplained incident",
        "a mystery story built around unanswered questions and gradual revelations",
    ],
    "Short": [
        "a movie poster for a short film",
        "a short film with a brief running time",
        "a short-format movie presenting a compact story or a focused cinematic idea",
        "a brief cinematic work released as a short rather than a full-length feature",
    ],
    "Romance": [
        "a movie poster for a romance film",
        "a romantic movie centered on love, attraction, and intimate relationships",
        "a film about people falling in love, separating, or trying to sustain a relationship",
        "a romance story exploring courtship, heartbreak, longing, or reconciliation",
    ],
    "Sci-Fi": [
        "a movie poster for a science fiction film",
        "a science fiction movie exploring imagined science, technology, or future societies",
        "a film involving space travel, alien life, time travel, robots, or advanced technology",
        "a science fiction story about the consequences of a scientific discovery or technological change",
    ],
    "Sport": [
        "a movie poster for a sports film",
        "a sports movie about athletes, teams, training, or competition",
        "a film about preparing for a sporting event and facing rivals or personal setbacks",
        "a story centered on athletic performance, teamwork, and the pursuit of sporting success",
    ],
    "Thriller": [
        "a movie poster for a thriller film",
        "a thriller driven by suspense, escalating danger, and uncertainty",
        "a film about characters caught in a pursuit, conspiracy, threat, or dangerous deception",
        "a tense story in which characters struggle to escape danger or prevent a looming disaster",
    ],
    "War": [
        "a movie poster for a war film",
        "a war movie about armed conflict and its effects on soldiers or civilians",
        "a film portraying military operations, battlefield survival, or life during wartime",
        "a story centered on the human experiences and consequences of war",
    ],
    "Western": [
        "a movie poster for a western film",
        "a western set on the American frontier with cowboys, settlers, or outlaws",
        "a film about frontier life, lawmen, ranchers, or conflicts in the Old West",
        "a western story featuring frontier towns, horseback travel, and struggles over land or justice",
    ],
}

LABEL_NAMES_GENRES = {genre: genre for genre in GENRES}


def get_prompt_set() -> dict:
    return PROMPTS_GENRES


def get_label_names() -> dict:
    return LABEL_NAMES_GENRES


def get_genre_list() -> list:
    return GENRES
