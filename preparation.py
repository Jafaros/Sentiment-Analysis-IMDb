from pathlib import Path

from utils.file import readFile
from utils.text import prepare_text

# Soubor pro načítání dat z IMDb datasetu a předzpracování textů
def load_reviews_from_folder(folder: Path, label: int):
	texts = []
	labels = []

	txt_files = sorted(folder.glob('*.txt'))
	for file_path in txt_files:
		raw_text = readFile(file_path)
		texts.append(prepare_text(raw_text))
		labels.append(label)

	return texts, labels

# Funkce pro načítání trénovacích a testovacích dat z IMDb datasetu
def load_imdb_splits(base_path: str = 'aclImdb'):
	root = Path(base_path)

	train_pos_texts, train_pos_labels = load_reviews_from_folder(root / 'train' / 'pos', 1)
	train_neg_texts, train_neg_labels = load_reviews_from_folder(root / 'train' / 'neg', 0)
	test_pos_texts, test_pos_labels = load_reviews_from_folder(root / 'test' / 'pos', 1)
	test_neg_texts, test_neg_labels = load_reviews_from_folder(root / 'test' / 'neg', 0)

	train_texts = train_pos_texts + train_neg_texts
	train_labels = train_pos_labels + train_neg_labels
	test_texts = test_pos_texts + test_neg_texts
	test_labels = test_pos_labels + test_neg_labels

	return {
		'train_texts': train_texts,
		'train_labels': train_labels,
		'test_texts': test_texts,
		'test_labels': test_labels,
	}
