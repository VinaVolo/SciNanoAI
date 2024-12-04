from transformers import pipeline

class ConductorAgent:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        # Используем предобученную модель для определения принадлежности к тематике
        self.classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        self.topics = ["Влияние наноструктурированной поверхности на поведение клеток", "Другая тема"]  # Замените на реальную тему

    def should_use_database(self, question):
        result = self.classifier(question, self.topics)
        scores = result['scores']
        labels = result['labels']
        # Предполагаем, что первый вариант в labels — это наиболее подходящая тема
        if labels[0] == self.topics[0] and scores[0] >= self.threshold:
            return True
        else:
            return False
