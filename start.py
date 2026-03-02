from utils.file import readFile
from pathlib import Path

from utils.text import prepare_text

base_path = Path('aclImdb')

folders_to_read = [
    base_path / 'train' / 'pos',
]

for folder in folders_to_read:
    txt_files = sorted(folder.glob('*.txt'))
    total_files = len(txt_files)

    if total_files == 0:
        print(f'No files found in: {folder}')
        continue

    print(f'Reading folder: {folder}')
    for index, file_path in enumerate(txt_files, start=1):
        print(f'Processed file: {index}/{total_files} - {file_path.name}')
        text = prepare_text(readFile(file_path))