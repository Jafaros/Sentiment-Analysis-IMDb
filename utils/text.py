import re

# Soubor pro funkce pro předzpracování textu
def prepare_text(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = remove_html_tags(cleaned)
    cleaned = remove_punctuation(cleaned)
    cleaned = normalize_spaces(cleaned)
    return cleaned

# Funkce pro odstranění interpunkce z textu
def remove_punctuation(text: str) -> str:
    return text.replace(',', '').replace('.', '').replace('!', '').replace('?', '').replace('"', '').replace("'", '').replace('-', ' ').replace('(', '').replace(')', '').replace(':', '').replace(';', '')

# Funkce pro odstranění HTML tagů z textu
def remove_html_tags(text: str) -> str:
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# Funkce pro normalizaci mezer v textu
def normalize_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()