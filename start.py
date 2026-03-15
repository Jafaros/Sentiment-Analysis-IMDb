import importlib.util
from pathlib import Path

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

    print(f'\nHotovo. Výstupy uloženy v: {output_dir.resolve()}')


if __name__ == '__main__':
    main()