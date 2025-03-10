# memory.py
# Gestion de la mémoire

import json
import logging
from config import MEMORY_PATH

logger = logging.getLogger("Memory")

class MemoryModule:
    def __init__(self):
        self.memory = self.load_memory()

    def load_memory(self):
        try:
            with open(MEMORY_PATH, 'r') as f:
                return json.load(f)
        except:
            return {}

    def save_memory(self):
        with open(MEMORY_PATH, 'w') as f:
            json.dump(self.memory, f, indent=4)

    def store_event(self, key, value):
        self.memory[key] = value
        self.save_memory()