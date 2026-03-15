# Soubor pro funkce pro načítání souborů s různými kódováními
def readFile(filePath: str) -> str:
    encodings = ('utf-8', 'cp1252', 'latin-1')

    for encoding in encodings:
        try:
            with open(filePath, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue

    with open(filePath, 'r', encoding='utf-8', errors='replace') as file:
        return file.read()