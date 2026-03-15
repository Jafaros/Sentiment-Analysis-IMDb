from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# TF-IDF + Logistic Regression klasifikátor pro sentimentální analýzu recenzí
class TfIdfClassifier:
    # Inicializace vektorizeru a modelu
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.model = LogisticRegression(max_iter=1000)

    # Metoda pro trénování modelu na trénovacích datech
    def fit(self, texts, labels):
        x_train = self.vectorizer.fit_transform(texts)
        self.model.fit(x_train, labels)
        return self

    # Metoda pro predikci sentimentu na základě textů
    def predict(self, texts):
        x = self.vectorizer.transform(texts)
        return self.model.predict(x)