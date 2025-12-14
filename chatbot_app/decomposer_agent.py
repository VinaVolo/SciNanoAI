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
            "photoresistors", "types of nanostructures", "machine learning in nanostructuring", "nanostructuring methods",
            "materials for two-photon polymerization", "the effect of nanostructuring on cells", "the effect of nanoparticles on cells",
            "nanotubes and nanopores in cellular technologies", "the effect of nanostructuring on cell adhesion",
            "the effect of nanostructuring on cell differentiation", "the effect of nanostructuring on cellular forces",
            "the effect of nanostructuring on cell migration", "the effect of nanostructuring on cell proliferation",
            "Another topic"
        ]

    def should_use_database(self, question):
        """
        Check if the question should be answered from the database.

        Args:
            question (str): The question to be answered.

        Returns:
            bool: True if the question should be answered from the database, False otherwise.
        """
        if not question or not question.strip():
            return False

        result = self.classifier(question, self.topics)
        top_label = result['labels'][0]
        top_score = result['scores'][0]

        if top_label == "Another topic" or top_score < self.threshold:
            return False
        return True
