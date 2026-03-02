import re

def prepare_text(text: str) -> str:
    return remove_html_tags(remove_punctuation(text.strip().lower()))

def remove_punctuation(text: str) -> str:
    return text.replace(',', '').replace('.', '').replace('!', '').replace('?', '').replace('"', '').replace("'", '').replace('-', ' ').replace('(', '').replace(')', '').replace(':', '').replace(';', '')

def remove_html_tags(text: str) -> str:
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)