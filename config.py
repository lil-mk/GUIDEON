# config.py
# Contient les paramètres globaux et constantes

import os

# Configuration des capteurs
USE_CAMERA = True
USE_MICROPHONE = True

# Paramètres réseau
CHECK_NETWORK = True

# Réglages du modèle IA
MODEL_PATH = "models/internal_model.json"
MEMORY_PATH = "models/self_awareness.json"

# Désactiver certaines optimisations de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'