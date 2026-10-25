import re
from html import unescape


def clean_text(text: str) -> str:
    """Use the same conservative cleanup during dataset preparation and inference."""
    text = unescape(text or "")
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def combine_news(headline: str, article: str) -> str:
    return f"{clean_text(headline)} {clean_text(article)}".strip()
