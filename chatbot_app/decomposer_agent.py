from transformers import pipeline

class DecomposerAgent:
    def __init__(self, threshold=0.2):
        """
        Initialize the DecomposerAgent.

        Args:
            threshold (float): The threshold above which the topic is considered relevant. Defaults to 0.2.
        """
        self.threshold = threshold
        self.classifier = pipeline('zero-shot-classification', model='BAAI/bge-reranker-large')
        self.topics = [
            "фоторезисторы", "типы наноструктур", "машинное обучение в наноструктурировании", "методы наноструктурирования",
            "материалы для двухфотонной полимеризации", "влияние наноструктурирования на клетки", "влияние наночастиц на клетки",
            "нанотрубки и нанопоры в клеточных технологиях", "влияние наноструктурирования на клеточную адгезию",
            "влияние наноструктурирования на дифференцирование клеток", "влияние наноструктурирования на клеточные силы",
            "влияние наноструктурирования на клеточную миграцию", "влияние наноструктурирования на клеточную пролиферацию",
            "Другая тема"
        ]

    def should_use_database(self, question):
        """
        Check if the question should be answered from the database.

        Args:
            question (str): The question to be answered.

        Returns:
            bool: True if the question should be answered from the database, False otherwise.
        """
        result = self.classifier(question, self.topics)
        top_label = result['labels'][0]
        top_score = result['scores'][0]

        if top_label == "Другая тема" or top_score < self.threshold:
            return False
        return True

