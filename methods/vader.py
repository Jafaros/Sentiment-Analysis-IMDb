from nltk.sentiment import SentimentIntensityAnalyzer

# VADER klasifikátor pro sentimentální analýzu recenzí
class VaderSentimentClassifier:
    # Inicializace analyzátoru
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    # Metoda pro získání compound score z textu
    def score(self, text: str) -> float:
        return self.analyzer.polarity_scores(text)['compound']

    # Metoda pro predikci sentimentu na základě compound score
    def predict(self, text: str) -> int:
        return 1 if self.score(text) >= 0 else 0

    # Metoda pro predikci sentimentu pro více textů a získání compound score pro každý text
    def predict_batch(self, texts):
        compounds = [self.score(text) for text in texts]
        predictions = [1 if compound >= 0 else 0 for compound in compounds]
        return predictions, compounds