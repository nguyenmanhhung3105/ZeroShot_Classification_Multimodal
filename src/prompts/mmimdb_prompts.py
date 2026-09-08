"""
Prompt templates cho MM-IMDb — phân loại thể loại phim (multi-label, 23 lớp).

LƯU Ý: đây là bài toán multi-label, KHÔNG dùng argmax như 2 dataset kia.
Mỗi genre có prompt riêng, khi encode xong sẽ dùng THRESHOLD (không phải argmax)
để quyết định 1 phim thuộc những genre nào — 1 phim có thể có 0, 1, hoặc nhiều genre.
"""

GENRES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History",
    "Horror", "Music", "Musical", "Mystery", "News", "Romance",
    "Sci-Fi", "Sport", "Thriller", "War", "Western",
]

# Mỗi genre có 2 câu prompt khác cách diễn đạt để hỗ trợ ensembling
PROMPTS_GENRES = {
    genre: [
        f"a movie poster for a {genre.lower()} film",
        f"a {genre.lower()} movie",
    ]
    for genre in GENRES
}

# Ghi đè riêng vài genre có tên khó diễn đạt tự nhiên nếu dùng template mặc định
PROMPTS_GENRES["Sci-Fi"] = [
    "a movie poster for a science fiction film",
    "a science fiction movie",
]
PROMPTS_GENRES["Film-Noir"] = [
    "a movie poster for a film-noir style movie",
    "a dark, moody crime film in film-noir style",
]

LABEL_NAMES_GENRES = {genre: genre for genre in GENRES}


def get_prompt_set() -> dict:
    return PROMPTS_GENRES


def get_label_names() -> dict:
    return LABEL_NAMES_GENRES


def get_genre_list() -> list:
    return GENRES
