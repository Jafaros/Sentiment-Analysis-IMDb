import importlib.util
from pathlib import Path
from shutil import copy2

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from methods.vader import VaderSentimentClassifier
from preparation import load_imdb_splits


# Dynamické načítání třídy TfIdfClassifier z modulu tf-idf.py
def load_tfidf_classifier_class():
    module_path = Path(__file__).parent / 'methods' / 'tf-idf.py'
    spec = importlib.util.spec_from_file_location('tf_idf_module', module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Unable to load tf-idf.py module spec.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TfIdfClassifier

# Funkce pro vyhodnocení metrik a zobrazení výsledků
def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='binary',
        pos_label=1,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
    }

# Funkce pro výpis metrik do konzole
def print_metrics(title, metrics):
    print(f'\n=== {title} ===')
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 skóre : {metrics['f1']:.4f}")
    print('Confusion matrix:')
    print(metrics['confusion_matrix'])

# Funkce pro vykreslení srovnávacího grafu metrik, Confusion matrix a histogramu compound score pro VADER
def plot_metrics_bar(vader_metrics, tfidf_metrics, output_dir: Path):
    labels = ['Accuracy', 'F1']
    vader_values = [vader_metrics['accuracy'], vader_metrics['f1']]
    tfidf_values = [tfidf_metrics['accuracy'], tfidf_metrics['f1']]

    x = [0, 1]
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([v - width / 2 for v in x], vader_values, width=width, label='VADER')
    plt.bar([v + width / 2 for v in x], tfidf_values, width=width, label='TF-IDF')
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel('Skóre')
    plt.title('Accuracy a F1: VADER vs TF-IDF')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=150)
    plt.close()

# Funkce pro vykreslení Confusion matrix a histogramu compound score pro VADER
def plot_confusion_matrix(cm, title, output_path: Path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    classes = ['negative', 'positive']
    ticks = [0, 1]
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)
    plt.xlabel('Předpokládaný label')
    plt.ylabel('Skutečný label')

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            color = 'white' if value > threshold else 'black'
            plt.text(j, i, str(value), ha='center', va='center', color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# Funkce pro vykreslení histogramu compound score pro VADER
def plot_compound_histogram(compounds, output_path: Path):
    plt.figure(figsize=(8, 5))
    plt.hist(compounds, bins=30, color='#1f77b4', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=1)
    plt.title('VADER Compound Score distribuce (test)')
    plt.xlabel('Compound score')
    plt.ylabel('Počet recenzí')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# Funkce pro získání cest k testovacím recenzím pro výběr chybně klasifikovaných příkladů
def get_test_review_paths(base_path: str = 'aclImdb'):
    root = Path(base_path)
    test_pos_paths = sorted((root / 'test' / 'pos').glob('*.txt'))
    test_neg_paths = sorted((root / 'test' / 'neg').glob('*.txt'))
    return test_pos_paths + test_neg_paths

# Funkce pro výběr dvou chybně klasifikovaných recenzí (1 false negative a 1 false positive) pro každý model
def select_two_misclassified_examples(test_paths, y_true, y_pred):
    false_negative_pos = None
    false_positive_neg = None

    for path, true_label, pred_label in zip(test_paths, y_true, y_pred):
        if true_label == 1 and pred_label == 0 and false_negative_pos is None:
            false_negative_pos = path
        elif true_label == 0 and pred_label == 1 and false_positive_neg is None:
            false_positive_neg = path

        if false_negative_pos is not None and false_positive_neg is not None:
            break

    if false_negative_pos is None or false_positive_neg is None:
        raise ValueError('Nepodarilo se najit 2 priklady (pos+neg) chybne klasifikovanych recenzi.')

    return {
        'false_negative_pos': false_negative_pos,
        'false_positive_neg': false_positive_neg,
    }

# Zkopírování dvou vybraných chybně klasifikovaných recenzí pro každý model do výstupní složky pro snadnou kontrolu a porovnání
def copy_selected_misclassified_reviews(output_dir: Path, test_paths, test_labels, vader_predictions, tfidf_predictions):
    selected_by_model = {
        'vader': select_two_misclassified_examples(test_paths, test_labels, vader_predictions),
        'tfidf': select_two_misclassified_examples(test_paths, test_labels, tfidf_predictions),
    }

    target_dir = output_dir / 'misclassified_reviews'
    target_dir.mkdir(parents=True, exist_ok=True)

    for old_file in target_dir.glob('vader_false_*.txt'):
        old_file.unlink()
    for old_file in target_dir.glob('tfidf_false_*.txt'):
        old_file.unlink()

    for model_name, selected_reviews in selected_by_model.items():
        for review_type, source_path in selected_reviews.items():
            target_name = f'{model_name}_{review_type}_{source_path.name}'
            copy2(source_path, target_dir / target_name)

# Hlavní funkce pro načítání dat, trénování modelů, vyhodnocení a ukládání výsledků
def main():
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    print('Načítání a příprava dat...')
    data = load_imdb_splits('aclImdb')

    train_texts = data['train_texts']
    train_labels = data['train_labels']
    test_texts = data['test_texts']
    test_labels = data['test_labels']
    test_paths = get_test_review_paths('aclImdb')

    if len(test_paths) != len(test_texts):
        raise RuntimeError('Pocet testovacich souboru neodpovida poctu testovacich textu.')

    print(f'Trénovací data: {len(train_texts)}')
    print(f'Testovací data: {len(test_texts)}')

    print('\nSpouštím VADER...')
    vader = VaderSentimentClassifier()
    vader_predictions, vader_compounds = vader.predict_batch(test_texts)
    vader_metrics = evaluate(test_labels, vader_predictions)
    print_metrics('VADER', vader_metrics)

    print('\nTrénuji TF-IDF + Logistic Regression...')
    TfIdfClassifier = load_tfidf_classifier_class()
    tfidf = TfIdfClassifier().fit(train_texts, train_labels)
    tfidf_predictions = tfidf.predict(test_texts)
    tfidf_metrics = evaluate(test_labels, tfidf_predictions)
    print_metrics('TF-IDF + Logistic Regression', tfidf_metrics)

    print('\nUkládám vizualizace a grafy...')
    plot_metrics_bar(vader_metrics, tfidf_metrics, output_dir)
    plot_confusion_matrix(vader_metrics['confusion_matrix'], 'Confusion Matrix - VADER', output_dir / 'confusion_matrix_vader.png')
    plot_confusion_matrix(tfidf_metrics['confusion_matrix'], 'Confusion Matrix - TF-IDF', output_dir / 'confusion_matrix_tfidf.png')
    plot_compound_histogram(vader_compounds, output_dir / 'vader_compound_histogram.png')
    copy_selected_misclassified_reviews(output_dir, test_paths, test_labels, vader_predictions, tfidf_predictions)

    print(f'\nHotovo. Výstupy uloženy v: {output_dir.resolve()}')


if __name__ == '__main__':
    main()