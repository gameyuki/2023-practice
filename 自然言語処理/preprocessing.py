import re

from bs4 import BeautifulSoup
from janome.tokenizer import Tokenizer
t = Tokenizer()

# hrmlテキストからタグを除去してテキストだけを返す
def clean_html(html, strip=False):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(strip=strip)
    return text

#　テキストを形態素解析して、単語ごとに分割する
def tokenize(text):
    return t.tokenize(text, wakati=True)

#　テキストを形態素解析して、単語の基本形を抽出してリストにして返す
def tokenize_base_from(text):
    tokens = [token.base_form for token in t.tokenize(text)]
    return tokens

# テキスト内の数字を正規化する
def normalize_number(text,reduce=False):
    if reduce:
        normalize_text = re.sub(r'\d+', '0', text)
    else:
        normalize_text = re.sub(r'\d', '0', text)
    return normalize_text