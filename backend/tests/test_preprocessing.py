from app.preprocessing.text import clean_text, combine_news


def test_clean_text_removes_html_and_replaces_urls():
    assert clean_text("<b>News</b> at https://example.org  today") == "News at URLTOKEN today"


def test_combine_news_keeps_headline_and_article():
    assert combine_news("A headline", "article text") == "A headline article text"
