# main.py
# Point d'entrée du programme

import logging
from sensors import SensorModule
from perception import PerceptionModule
from nlp import NLPModule
from decision_making import DecisionModule
from memory import MemoryModule
from ai_engine import AIEngine
from config import USE_CAMERA, USE_MICROPHONE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GIDEON")

class GIDEONSystem:
    def __init__(self):
        self.sensor_module = SensorModule(USE_CAMERA, USE_MICROPHONE)
        self.perception_module = PerceptionModule()
        self.nlp_module = NLPModule()
        self.decision_module = DecisionModule()
        self.memory_module = MemoryModule()
        self.ai_engine = AIEngine()
        self.running = False

    def start(self):
        logger.info("Système GIDEON démarré")
        self.running = True
        while self.running:
            text = self.sensor_module.listen()
            if text:
                entities = self.nlp_module.analyze_text(text)
                decision = self.decision_module.make_decision(text)
                logger.info(f"Utilisateur: {text} | Décision: {decision} | Entités: {entities}")
                self.memory_module.store_event("last_interaction", text)
                logger.info(self.ai_engine.train_model([text]))

    def stop(self):
        self.running = False
        self.sensor_module.release_resources()
        logger.info("Système GIDEON arrêté")

if __name__ == "__main__":
    system = GIDEONSystem()
    system.start()
