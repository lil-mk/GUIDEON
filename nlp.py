# nlp.py
# Traitement du langage naturel

import spacy
import logging

logger = logging.getLogger("NLP")

class NLPModule:
    def __init__(self):
        try:
            self.nlp = spacy.load("fr_core_news_md")
        except:
            self.nlp = None

    def analyze_text(self, text):
        if self.nlp:
            doc = self.nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]
        return []