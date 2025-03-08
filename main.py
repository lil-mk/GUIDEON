"""
Système d'Auto-Perception pour IA
=================================
Ce programme implémente un système d'auto-perception pour une IA,
créant un modèle interne de ses capteurs et de son environnement.
"""

import cv2
import numpy as np
import time
import threading
import speech_recognition as sr
import pyttsx3
from collections import deque
import json
import os
import logging
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Conv2D, MaxPooling2D, Flatten, Concatenate

# Configuration du système de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SelfAwareSystem')


class SensorModule:
    """Module gérant les capteurs (caméra, microphone) et leur traitement"""

    def __init__(self, use_camera=True, use_microphone=True):
        self.use_camera = use_camera
        self.use_microphone = use_microphone
        self.camera = None
        self.recognizer = None
        self.microphone = None
        self.engine = None

        # Données sensorielles actuelles
        self.current_frame = None
        self.current_audio = None
        self.last_recognized_speech = ""

        # Historique des données sensorielles
        self.visual_history = deque(maxlen=30)  # ~1s à 30 FPS
        self.audio_history = deque(maxlen=5)  # Dernières 5 commandes vocales

        # État des capteurs
        self.sensor_states = {
            "camera": {"active": False, "reliability": 1.0, "last_update": 0},
            "microphone": {"active": False, "reliability": 1.0, "last_update": 0}
        }

        # Initialisation des capteurs
        self._initialize_sensors()

    def _initialize_sensors(self):
        """Initialise les capteurs disponibles"""
        try:
            if self.use_camera:
                self.camera = cv2.VideoCapture(0)
                if self.camera.isOpened():
                    self.sensor_states["camera"]["active"] = True
                    logger.info("Caméra initialisée avec succès")
                else:
                    logger.warning("Échec d'initialisation de la caméra")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la caméra: {e}")

        try:
            if self.use_microphone:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                self.engine = pyttsx3.init()
                self.sensor_states["microphone"]["active"] = True
                logger.info("Microphone initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du microphone: {e}")

    def capture_frame(self):
        """Capture une image depuis la caméra"""
        if not self.sensor_states["camera"]["active"]:
            return None

        ret, frame = self.camera.read()
        if ret:
            self.current_frame = frame
            self.visual_history.append(frame)
            self.sensor_states["camera"]["last_update"] = time.time()
            return frame
        else:
            self.sensor_states["camera"]["reliability"] -= 0.1
            self.sensor_states["camera"]["reliability"] = max(0.0, self.sensor_states["camera"]["reliability"])
            logger.warning("Échec de capture d'image")
            return None

    def listen(self, timeout=5):
        """Capte l'audio depuis le microphone"""
        if not self.sensor_states["microphone"]["active"]:
            return ""

        try:
            with self.microphone as source:
                logger.info("En écoute...")
                audio = self.recognizer.listen(source, timeout=timeout)
                self.current_audio = audio
                self.sensor_states["microphone"]["last_update"] = time.time()

                try:
                    text = self.recognizer.recognize_google(audio, language="fr-FR")
                    logger.info(f"Texte reconnu: {text}")
                    self.last_recognized_speech = text
                    self.audio_history.append(text)
                    return text
                except sr.UnknownValueError:
                    logger.info("Reconnaissance vocale n'a pas compris l'audio")
                    return ""
                except sr.RequestError as e:
                    logger.error(f"Erreur de service: {e}")
                    self.sensor_states["microphone"]["reliability"] -= 0.1
                    self.sensor_states["microphone"]["reliability"] = max(0.0, self.sensor_states["microphone"][
                        "reliability"])
                    return ""
        except Exception as e:
            logger.error(f"Erreur lors de l'écoute: {e}")
            return ""

    def speak(self, text):
        """Utilise la synthèse vocale pour parler"""
        if self.engine:
            try:
                logger.info(f"Synthèse vocale: {text}")
                self.engine.say(text)
                self.engine.runAndWait()
                return True
            except Exception as e:
                logger.error(f"Erreur lors de la synthèse vocale: {e}")
                return False
        return False

    def get_sensor_states(self):
        """Renvoie l'état actuel des capteurs"""
        # Mettre à jour les états de fiabilité basés sur le temps écoulé
        current_time = time.time()
        for sensor in self.sensor_states:
            time_since_update = current_time - self.sensor_states[sensor]["last_update"]
            if time_since_update > 10 and self.sensor_states[sensor]["active"]:  # 10 secondes sans mise à jour
                self.sensor_states[sensor]["reliability"] -= 0.05
                self.sensor_states[sensor]["reliability"] = max(0.0, self.sensor_states[sensor]["reliability"])

        return self.sensor_states

    def release_resources(self):
        """Libère les ressources utilisées par les capteurs"""
        if self.camera and self.camera.isOpened():
            self.camera.release()
        if self.engine:
            self.engine.stop()
        logger.info("Ressources des capteurs libérées")


class SelfModel:
    """Module gérant le modèle interne du système (modèle de soi)"""

    def __init__(self, sensor_module):
        self.sensor_module = sensor_module
        self.internal_model = {
            "sensors": {
                "camera": {
                    "type": "visual",
                    "field_of_view": 60,  # en degrés, approximation
                    "resolution": (640, 480),
                    "position": [0, 0, 0],  # position relative x, y, z
                    "reliability": 1.0,
                    "active": True
                },
                "microphone": {
                    "type": "audio",
                    "sensitivity": 0.8,
                    "noise_floor": 0.1,
                    "position": [0, 0, 0],
                    "reliability": 1.0,
                    "active": True
                }
            },
            "environment": {
                "detected_objects": [],
                "audio_events": [],
                "last_interaction": "",
                "ambient_conditions": {
                    "light_level": 0.5,  # normalisé entre 0 et 1
                    "noise_level": 0.2  # normalisé entre 0 et 1
                }
            },
            "system_state": {
                "attention_focus": None,
                "confidence": 0.9,
                "last_update": time.time()
            }
        }
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.training_data = []

        # Création du dossier pour sauvegarder le modèle
        os.makedirs("model_data", exist_ok=True)

    def update_sensor_data(self):
        """Met à jour le modèle interne avec les données des capteurs"""
        sensor_states = self.sensor_module.get_sensor_states()

        # Mise à jour de l'état des capteurs dans le modèle interne
        for sensor in sensor_states:
            if sensor in self.internal_model["sensors"]:
                self.internal_model["sensors"][sensor]["reliability"] = sensor_states[sensor]["reliability"]
                self.internal_model["sensors"][sensor]["active"] = sensor_states[sensor]["active"]

        # Analyse de l'image si disponible
        if self.sensor_module.current_frame is not None:
            # Exemple simple: détection de luminosité
            frame = self.sensor_module.current_frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            light_level = np.mean(gray) / 255.0
            self.internal_model["environment"]["ambient_conditions"]["light_level"] = light_level

            # Ici, on pourrait ajouter la détection d'objets avec un modèle pré-entraîné
            # Par exemple, avec YOLO ou un autre détecteur d'objets

        # Mise à jour des données audio
        if self.sensor_module.last_recognized_speech:
            self.internal_model["environment"]["last_interaction"] = self.sensor_module.last_recognized_speech

        # Mise à jour de l'horodatage
        self.internal_model["system_state"]["last_update"] = time.time()

        # Collecter des données pour la détection d'anomalies
        feature_vector = [
            self.internal_model["sensors"]["camera"]["reliability"],
            self.internal_model["sensors"]["microphone"]["reliability"],
            self.internal_model["environment"]["ambient_conditions"]["light_level"],
            self.internal_model["environment"]["ambient_conditions"]["noise_level"]
        ]
        self.training_data.append(feature_vector)

        # Entraîner le détecteur d'anomalies si suffisamment de données
        if len(self.training_data) >= 100:
            self._train_anomaly_detector()

    def _train_anomaly_detector(self):
        """Entraîne le détecteur d'anomalies sur les données collectées"""
        try:
            X = np.array(self.training_data[-100:])  # Utiliser les 100 dernières observations
            self.anomaly_detector.fit(X)
            logger.info("Détecteur d'anomalies entraîné avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement du détecteur d'anomalies: {e}")

    def detect_anomalies(self):
        """Détecte les anomalies dans l'état actuel du système"""
        if len(self.training_data) < 100:
            return False  # Pas assez de données pour une détection fiable

        current_state = [
            self.internal_model["sensors"]["camera"]["reliability"],
            self.internal_model["sensors"]["microphone"]["reliability"],
            self.internal_model["environment"]["ambient_conditions"]["light_level"],
            self.internal_model["environment"]["ambient_conditions"]["noise_level"]
        ]

        try:
            anomaly_score = self.anomaly_detector.decision_function([current_state])[0]
            is_anomaly = anomaly_score < -0.2  # Seuil arbitraire

            if is_anomaly:
                logger.warning(f"Anomalie détectée! Score: {anomaly_score}")
                # Mettre à jour la confiance du système
                self.internal_model["system_state"]["confidence"] -= 0.1
                self.internal_model["system_state"]["confidence"] = max(0.1, self.internal_model["system_state"][
                    "confidence"])
            else:
                # Restaurer progressivement la confiance
                self.internal_model["system_state"]["confidence"] += 0.01
                self.internal_model["system_state"]["confidence"] = min(1.0, self.internal_model["system_state"][
                    "confidence"])

            return is_anomaly
        except Exception as e:
            logger.error(f"Erreur lors de la détection d'anomalies: {e}")
            return False

    def set_attention_focus(self, focus_target):
        """Définit la cible d'attention actuelle du système"""
        self.internal_model["system_state"]["attention_focus"] = focus_target
        logger.info(f"Attention focalisée sur: {focus_target}")

    def save_model(self, filename="model_data/internal_model.json"):
        """Sauvegarde le modèle interne dans un fichier JSON"""
        try:
            # Créer une copie du modèle pour la sérialisation
            serializable_model = {k: v for k, v in self.internal_model.items()}

            with open(filename, 'w') as f:
                json.dump(serializable_model, f, indent=4)

            logger.info(f"Modèle interne sauvegardé dans {filename}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle: {e}")
            return False

    def load_model(self, filename="model_data/internal_model.json"):
        """Charge le modèle interne depuis un fichier JSON"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    loaded_model = json.load(f)

                # Mettre à jour le modèle interne avec les données chargées
                for key in loaded_model:
                    if key in self.internal_model:
                        self.internal_model[key] = loaded_model[key]

                logger.info(f"Modèle interne chargé depuis {filename}")
                return True
            else:
                logger.warning(f"Fichier de modèle {filename} non trouvé")
                return False
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            return False

    def get_self_representation(self):
        """Renvoie une représentation textuelle du modèle de soi"""
        sensors_status = []
        for sensor_name, sensor_data in self.internal_model["sensors"].items():
            status = "actif" if sensor_data["active"] else "inactif"
            reliability = int(sensor_data["reliability"] * 100)
            sensors_status.append(f"{sensor_name} ({status}, fiabilité: {reliability}%)")

        attention = self.internal_model["system_state"]["attention_focus"] or "aucune"
        confidence = int(self.internal_model["system_state"]["confidence"] * 100)

        representation = f"État actuel du système:\n"
        representation += f"- Capteurs: {', '.join(sensors_status)}\n"
        representation += f"- Attention focalisée sur: {attention}\n"
        representation += f"- Niveau de confiance: {confidence}%\n"
        representation += f"- Dernière mise à jour: {time.ctime(self.internal_model['system_state']['last_update'])}"

        return representation


class LearningModule:
    """Module d'apprentissage pour l'adaptation du système"""

    def __init__(self, model_dir="model_data"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Modèle pour la prédiction des états futurs
        self.prediction_model = None
        self.initialized = False

        # Mémoire d'expérience
        self.experience_buffer = deque(maxlen=1000)

        # Métadonnées d'apprentissage
        self.learning_stats = {
            "training_iterations": 0,
            "last_loss": 0,
            "observations": 0
        }

    def initialize_models(self):
        """Initialise ou charge les modèles d'apprentissage"""
        try:
            model_path = os.path.join(self.model_dir, "prediction_model.h5")

            if os.path.exists(model_path):
                logger.info("Chargement du modèle de prédiction existant")
                self.prediction_model = load_model(model_path)
            else:
                logger.info("Création d'un nouveau modèle de prédiction")
                # Modèle simple pour la prédiction
                self._create_prediction_model()

            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des modèles: {e}")
            return False

    def _create_prediction_model(self):
        """Crée un modèle de prédiction basé sur LSTM"""
        # Architecture du modèle
        # Entrée: séquence d'états [fiabilité_caméra, fiabilité_micro, niveau_lumière, niveau_bruit]
        # Sortie: prédiction de l'état suivant

        input_dim = 4  # Nombre de caractéristiques dans chaque état
        sequence_length = 10  # Nombre d'états précédents à considérer

        model = Sequential([
            LSTM(64, input_shape=(sequence_length, input_dim), return_sequences=True),
            LSTM(32),
            Dense(32, activation='relu'),
            Dense(input_dim)  # Prédiction de l'état suivant
        ])

        model.compile(optimizer='adam', loss='mse')
        self.prediction_model = model
        logger.info("Modèle de prédiction créé")

    def record_experience(self, current_state, action, next_state, reward):
        """Enregistre une expérience dans la mémoire d'apprentissage"""
        experience = {
            "current_state": current_state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "timestamp": time.time()
        }

        self.experience_buffer.append(experience)
        self.learning_stats["observations"] += 1
        logger.debug(f"Expérience enregistrée, total: {self.learning_stats['observations']}")

    def train_prediction_model(self, batch_size=32):
        """Entraîne le modèle de prédiction sur les expériences accumulées"""
        if not self.initialized:
            logger.warning("Modèle non initialisé, impossible d'entraîner")
            return False

        if len(self.experience_buffer) < batch_size:
            logger.info(f"Pas assez d'expériences pour l'entraînement ({len(self.experience_buffer)}/{batch_size})")
            return False

        try:
            # Préparer les données d'entraînement
            states_sequences = []
            next_states = []

            # Parcourir le buffer pour créer des séquences
            for i in range(len(self.experience_buffer) - 10):
                sequence = [exp["current_state"] for exp in list(self.experience_buffer)[i:i + 10]]
                target = self.experience_buffer[i + 10]["next_state"]

                if len(sequence) == 10:  # Vérifier la longueur de la séquence
                    states_sequences.append(sequence)
                    next_states.append(target)

            if not states_sequences:
                return False

            # Convertir en tableaux numpy
            X = np.array(states_sequences)
            y = np.array(next_states)

            # Entraîner le modèle
            history = self.prediction_model.fit(
                X, y,
                epochs=5,
                batch_size=min(batch_size, len(X)),
                verbose=0
            )

            # Mettre à jour les statistiques
            self.learning_stats["training_iterations"] += 1
            self.learning_stats["last_loss"] = history.history['loss'][-1]

            logger.info(f"Entraînement terminé, perte: {self.learning_stats['last_loss']}")

            # Sauvegarder le modèle périodiquement
            if self.learning_stats["training_iterations"] % 10 == 0:
                self.save_models()

            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {e}")
            return False

    def predict_next_state(self, state_history):
        """Prédit l'état suivant à partir de l'historique des états"""
        if not self.initialized or len(state_history) < 10:
            return None

        try:
            # Préparer la séquence d'entrée
            sequence = np.array([state_history[-10:]])

            # Faire la prédiction
            prediction = self.prediction_model.predict(sequence)
            return prediction[0]
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            return None

    def save_models(self):
        """Sauvegarde les modèles d'apprentissage"""
        if not self.initialized:
            return False

        try:
            model_path = os.path.join(self.model_dir, "prediction_model.h5")
            self.prediction_model.save(model_path)

            # Sauvegarder les statistiques d'apprentissage
            stats_path = os.path.join(self.model_dir, "learning_stats.json")
            with open(stats_path, 'w') as f:
                json.dump(self.learning_stats, f, indent=4)

            logger.info("Modèles et statistiques d'apprentissage sauvegardés")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des modèles: {e}")
            return False

    def load_stats(self):
        """Charge les statistiques d'apprentissage"""
        stats_path = os.path.join(self.model_dir, "learning_stats.json")

        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    self.learning_stats = json.load(f)
                logger.info("Statistiques d'apprentissage chargées")
                return True
            except Exception as e:
                logger.error(f"Erreur lors du chargement des statistiques: {e}")
                return False
        return False

    def get_learning_status(self):
        """Renvoie un résumé du statut d'apprentissage"""
        return {
            "initialized": self.initialized,
            "experiences": len(self.experience_buffer),
            "training_iterations": self.learning_stats["training_iterations"],
            "last_loss": self.learning_stats["last_loss"],
            "total_observations": self.learning_stats["observations"]
        }


class CommunicationInterface:
    """Interface de communication entre le système et l'utilisateur"""

    def __init__(self, sensor_module, self_model):
        self.sensor_module = sensor_module
        self.self_model = self_model
        self.conversation_history = deque(maxlen=20)
        self.context = {}

        # Phrases préfabriquées pour la communication
        self.responses = {
            "greeting": [
                "Bonjour, je suis votre assistant IA auto-perceptif.",
                "Salut! Je suis en ligne et j'écoute.",
                "Bonjour, je suis prêt à interagir avec vous."
            ],
            "farewell": [
                "Au revoir, à bientôt!",
                "À la prochaine fois!",
                "Je me mets en veille. À bientôt!"
            ],
            "confirmation": [
                "D'accord, c'est noté.",
                "Compris.",
                "J'ai bien enregistré cette information."
            ],
            "confusion": [
                "Je n'ai pas bien compris, pourriez-vous reformuler?",
                "Désolé, je n'ai pas saisi votre demande.",
                "Pourriez-vous préciser ce que vous attendez de moi?"
            ],
            "self_description": [
                "Je suis un système d'IA doté de capacités d'auto-perception. Je peux voir, entendre et maintenir un modèle interne de mes capteurs.",
                "Je suis conçu pour être conscient de mes propres capteurs et de leur état.",
                "Je dispose d'une caméra et d'un microphone qui me permettent d'interagir avec mon environnement."
            ]
        }

    def process_input(self, text_input):
        """Traite une entrée textuelle et génère une réponse appropriée"""
        if not text_input:
            return self.get_random_response("confusion")

        # Enregistrer l'interaction dans l'historique
        self.conversation_history.append({"role": "user", "content": text_input})

        # Analyse simple des commandes (à améliorer avec NLP)
        text_lower = text_input.lower()

        # Réponses basées sur l'entrée
        if any(greeting in text_lower for greeting in ["bonjour", "salut", "hello", "coucou"]):
            response = self.get_random_response("greeting")

        elif any(farewell in text_lower for farewell in ["au revoir", "adieu", "bye", "à plus"]):
            response = self.get_random_response("farewell")

        elif any(self_query in text_lower for self_query in ["qui es-tu", "présente-toi", "tes capacités"]):
            response = self.get_random_response("self_description")

        elif "état" in text_lower or "status" in text_lower:
            # Demande d'état du système
            response = self.self_model.get_self_representation()

        elif "caméra" in text_lower or "image" in text_lower or "voir" in text_lower:
            # Commande liée à la vision
            if "active" in text_lower or "allume" in text_lower:
                self.self_model.internal_model["sensors"]["camera"]["active"] = True
                response = "J'ai activé ma caméra."
            elif "désactive" in text_lower or "éteins" in text_lower:
                self.self_model.internal_model["sensors"]["camera"]["active"] = False
                response = "J'ai désactivé ma caméra."
            else:
                if self.self_model.internal_model["sensors"]["camera"]["active"]:
                    response = "Ma caméra est active et je peux voir."
                else:
                    response = "Ma caméra est actuellement désactivée."

        elif "micro" in text_lower or "écoute" in text_lower or "entendre" in text_lower:
            # Commande liée à l'audio
            if "active" in text_lower or "allume" in text_lower:
                self.self_model.internal_model["sensors"]["microphone"]["active"] = True
                response = "J'ai activé mon microphone."
            elif "désactive" in text_lower or "éteins" in text_lower:
                self.self_model.internal_model["sensors"]["microphone"]["active"] = False
                response = "J'ai désactivé mon microphone."
            else:
                if self.self_model.internal_model["sensors"]["microphone"]["active"]:
                    response = "Mon microphone est actif et je peux vous entendre."
                else:
                    response = "Mon microphone est actuellement désactivé."

        elif "sauvegarde" in text_lower:
            # Sauvegarde du modèle interne
            success = self.self_model.save_model()
            if success:
                response = "J'ai sauvegardé mon modèle interne avec succès."
            else:
                response = "Il y a eu un problème lors de la sauvegarde de mon modèle."

        elif "charge" in text_lower or "restaure" in text_lower:
            # Chargement du modèle interne
            success = self.self_model.load_model()
            if success:
                response = "J'ai chargé mon modèle interne avec succès."
            else:
                response = "Il y a eu un problème lors du chargement de mon modèle."

        else:
            # Réponse générique
            response = "J'ai bien reçu votre message, mais je ne suis pas sûr de comprendre la demande."

        # Enregistrer la réponse dans l'historique
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def get_random_response(self, category):
        """Renvoie une réponse aléatoire d'une catégorie donnée"""
        if category in self.responses:
            return np.random.choice(self.responses[category])
        return "Je ne sais pas quoi répondre à cela."

    def speak_response(self, response):
        """Utilise la synthèse vocale pour énoncer la réponse"""
        return self.sensor_module.speak(response)

    def get_conversation_history(self):
        """Renvoie l'historique des conversations"""
        return list(self.conversation_history)

    def clear_history(self):
        """Efface l'historique des conversations"""
        self.conversation_history.clear()
        logger.info("Historique des conversations effacé")


class SelfAwareSystem:
    """Classe principale intégrant tous les modules du système auto-perceptif"""

    def __init__(self, use_camera=True, use_microphone=True):
        logger.info("Initialisation du système d'auto-perception...")

        # Initialisation des modules
        self.sensor_module = SensorModule(use_camera, use_microphone)
        self.self_model = SelfModel(self.sensor_module)
        self.learning_module = LearningModule()
        self.communication = CommunicationInterface(self.sensor_module, self.self_model)

        # État du système
        self.running = False
        self.state_history = deque(maxlen=100)

        # Threads pour les différentes tâches
        self.threads = {}
        self.threads = {
            "perception": None,
            "learning": None,
            "communication": None
        }

        # Initialiser le modèle d'apprentissage
        self.learning_module.initialize_models()
        self.learning_module.load_stats()

        logger.info("Système d'auto-perception initialisé")

    def start(self):
        """Démarre le système et ses threads"""
        if self.running:
            logger.warning("Le système est déjà en cours d'exécution")
            return False

        try:
            # Charger le modèle interne s'il existe
            self.self_model.load_model()

            # Démarrer les threads
            self.running = True

            # Thread de perception
            self.threads["perception"] = threading.Thread(
                target=self._perception_loop,
                daemon=True
            )
            self.threads["perception"].start()

            # Thread d'apprentissage
            self.threads["learning"] = threading.Thread(
                target=self._learning_loop,
                daemon=True
            )
            self.threads["learning"].start()

            # Thread de communication
            self.threads["communication"] = threading.Thread(
                target=self._communication_loop,
                daemon=True
            )
            self.threads["communication"].start()

            logger.info("Tous les threads du système ont été démarrés")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du démarrage du système: {e}")
            self.running = False
            return False

    def stop(self):
        """Arrête le système et libère les ressources"""
        if not self.running:
            logger.warning("Le système n'est pas en cours d'exécution")
            return

        try:
            # Arrêter les threads
            self.running = False

            # Attendre que les threads se terminent (avec timeout)
            for thread_name, thread in self.threads.items():
                if thread and thread.is_alive():
                    logger.info(f"Attente de la terminaison du thread {thread_name}...")
                    thread.join(timeout=2)

            # Sauvegarder l'état du système
            self.self_model.save_model()
            self.learning_module.save_models()

            # Libérer les ressources
            self.sensor_module.release_resources()

            logger.info("Système arrêté avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt du système: {e}")

    def _perception_loop(self):
        """Boucle principale de perception qui s'exécute dans un thread séparé"""
        logger.info("Boucle de perception démarrée")
        while self.running:
            try:
                # Capturer les données sensorielles
                if self.self_model.internal_model["sensors"]["camera"]["active"]:
                    self.sensor_module.capture_frame()

                # Mettre à jour le modèle interne
                self.self_model.update_sensor_data()

                # Détecter les anomalies
                anomaly_detected = self.self_model.detect_anomalies()
                if anomaly_detected:
                    logger.warning("Anomalie détectée dans l'état du système")

                # Enregistrer l'état actuel
                current_state = [
                    self.self_model.internal_model["sensors"]["camera"]["reliability"],
                    self.self_model.internal_model["sensors"]["microphone"]["reliability"],
                    self.self_model.internal_model["environment"]["ambient_conditions"]["light_level"],
                    self.self_model.internal_model["environment"]["ambient_conditions"]["noise_level"]
                ]
                self.state_history.append(current_state)

                # Attendre un peu avant la prochaine itération
                time.sleep(0.1)  # 10 Hz
            except Exception as e:
                logger.error(f"Erreur dans la boucle de perception: {e}")
                time.sleep(1)  # Attendre un peu plus longtemps en cas d'erreur

    def _learning_loop(self):
        """Boucle d'apprentissage qui s'exécute dans un thread séparé"""
        logger.info("Boucle d'apprentissage démarrée")
        learning_interval = 10  # Intervalle d'apprentissage en secondes
        last_learning_time = 0

        while self.running:
            try:
                current_time = time.time()

                # Vérifier s'il est temps d'apprendre
                if current_time - last_learning_time > learning_interval and len(self.state_history) >= 11:
                    # Créer une expérience à partir des états récents
                    current_state = self.state_history[-2]
                    next_state = self.state_history[-1]

                    # Action factice pour l'instant (pourrait être amélioré)
                    action = "observe"

                    # Récompense basée sur la fiabilité des capteurs
                    reward = (current_state[0] + current_state[1]) / 2

                    # Enregistrer l'expérience
                    self.learning_module.record_experience(
                        current_state, action, next_state, reward
                    )

                    # Entraîner le modèle si assez d'expériences
                    if len(self.learning_module.experience_buffer) >= 50:
                        logger.info("Démarrage d'une session d'apprentissage...")
                        self.learning_module.train_prediction_model()

                    last_learning_time = current_time

                # Faire des prédictions périodiquement
                if len(self.state_history) >= 10 and self.learning_module.initialized:
                    state_seq = list(self.state_history)[-10:]
                    predicted_next_state = self.learning_module.predict_next_state(state_seq)

                    if predicted_next_state is not None:
                        # On pourrait utiliser cette prédiction pour anticiper des problèmes
                        predicted_camera_reliability = predicted_next_state[0]
                        if predicted_camera_reliability < 0.5:
                            logger.warning("Prédiction: la fiabilité de la caméra pourrait diminuer")

                time.sleep(1)
            except Exception as e:
                logger.error(f"Erreur dans la boucle d'apprentissage: {e}")
                time.sleep(5)  # Attendre plus longtemps en cas d'erreur

    def _communication_loop(self):
        """Boucle de communication qui s'exécute dans un thread séparé"""
        logger.info("Boucle de communication démarrée")

        # Message de bienvenue
        welcome_message = "Système d'auto-perception démarré. Je suis prêt à interagir."
        self.communication.speak_response(welcome_message)

        while self.running:
            try:
                # Attendre une commande vocale si le microphone est actif
                if self.self_model.internal_model["sensors"]["microphone"]["active"]:
                    speech_input = self.sensor_module.listen(timeout=3)

                    if speech_input:
                        logger.info(f"Entrée vocale reçue: {speech_input}")

                        # Traiter l'entrée et générer une réponse
                        response = self.communication.process_input(speech_input)

                        # Énoncer la réponse
                        self.communication.speak_response(response)

                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Erreur dans la boucle de communication: {e}")
                time.sleep(2)

    def get_system_status(self):
        """Renvoie un résumé complet de l'état du système"""
        status = {
            "running": self.running,
            "sensors": self.sensor_module.get_sensor_states(),
            "self_model": {
                "confidence": self.self_model.internal_model["system_state"]["confidence"],
                "attention_focus": self.self_model.internal_model["system_state"]["attention_focus"],
                "last_update": self.self_model.internal_model["system_state"]["last_update"]
            },
            "learning": self.learning_module.get_learning_status(),
            "communication": {
                "history_length": len(self.communication.conversation_history)
            }
        }
        return status



def initialize(self):
        """Initialise tous les composants du système"""
        success = True
        
        # Charger le modèle interne s'il existe
        if os.path.exists("model_data/internal_model.json"):
            success = success and self.self_model.load_model()
        
        # Initialiser les modèles d'apprentissage
        success = success and self.learning_module.initialize_models()
        self.learning_module.load_stats()
        
        logger.info(f"Initialisation du système {'réussie' if success else 'incomplète'}")
        return success


def start(self):
        """Démarre le système et ses différents threads"""
        if self.running:
            logger.warning("Le système est déjà en cours d'exécution")
            return False
        
        self.running = True
        
        # Thread pour la perception
        self.threads["perception"] = threading.Thread(target=self._perception_loop)
        self.threads["perception"].daemon = True
        self.threads["perception"].start()
        
        # Thread pour l'apprentissage
        self.threads["learning"] = threading.Thread(target=self._learning_loop)
        self.threads["learning"].daemon = True
        self.threads["learning"].start()
        
        # Thread pour l'écoute vocale
        if self.sensor_module.sensor_states["microphone"]["active"]:
            self.threads["voice"] = threading.Thread(target=self._voice_interaction_loop)
            self.threads["voice"].daemon = True
            self.threads["voice"].start()
        
        logger.info("Système démarré avec succès")
        
        # Message de bienvenue
        welcome = self.communication.get_random_response("greeting")
        self.communication.speak_response(welcome)
        
        return True
    
def stop(self):
        """Arrête le système et libère les ressources"""
        if not self.running:
            logger.warning("Le système n'est pas en cours d'exécution")
            return False
        
        self.running = False
        
        # Attendre que les threads se terminent
        for thread_name, thread in self.threads.items():
            if thread.is_alive():
                logger.info(f"Attente de la fin du thread {thread_name}...")
                thread.join(timeout=2)
        
        # Sauvegarder l'état du système
        self.self_model.save_model()
        self.learning_module.save_models()
        
        # Libérer les ressources
        self.sensor_module.release_resources()
        
        logger.info("Système arrêté avec succès")
        return True
    
def _perception_loop(self):
        """Boucle principale pour la perception et la mise à jour du modèle interne"""
        logger.info("Démarrage de la boucle de perception")
        
        while self.running:
            try:
                # Capturer une image si la caméra est active
                if self.self_model.internal_model["sensors"]["camera"]["active"]:
                    self.sensor_module.capture_frame()
                
                # Mettre à jour le modèle interne
                self.self_model.update_sensor_data()
                
                # Enregistrer l'état actuel
                current_state = [
                    self.self_model.internal_model["sensors"]["camera"]["reliability"],
                    self.self_model.internal_model["sensors"]["microphone"]["reliability"],
                    self.self_model.internal_model["environment"]["ambient_conditions"]["light_level"],
                    self.self_model.internal_model["environment"]["ambient_conditions"]["noise_level"]
                ]
                self.state_history.append(current_state)
                
                # Détecter les anomalies
                self.self_model.detect_anomalies()
                
                # Pause pour économiser les ressources
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Erreur dans la boucle de perception: {e}")
                time.sleep(1)  # Pause plus longue en cas d'erreur

def _learning_loop(self):
        """Boucle principale pour l'apprentissage"""
        logger.info("Démarrage de la boucle d'apprentissage")
        
        # Attendre d'avoir suffisamment de données
        time.sleep(10)
        
        while self.running:
            try:
                if len(self.state_history) >= 10:
                    # Prédire l'état suivant
                    current_sequence = list(self.state_history)[-10:]
                    predicted_next_state = self.learning_module.predict_next_state(current_sequence)
                    
                    if predicted_next_state is not None and len(self.state_history) > 10:
                        # Calculer l'erreur de prédiction
                        actual_next_state = self.state_history[-1]
                        prediction_error = np.mean(np.abs(np.array(predicted_next_state) - np.array(actual_next_state)))
                        
                        # Récompense basée sur la précision de la prédiction
                        reward = 1.0 - min(1.0, prediction_error)
                        
                        # Enregistrer l'expérience
                        self.learning_module.record_experience(
                            current_sequence[-2],  # État actuel
                            None,  # Pas d'action explicite dans ce cas
                            actual_next_state,  # État suivant réel
                            reward  # Récompense
                        )
                
                # Entraîner le modèle périodiquement
                if self.learning_module.learning_stats["observations"] % 100 == 0 and self.learning_module.learning_stats["observations"] > 0:
                    self.learning_module.train_prediction_model()
                
                # Pause entre les cycles d'apprentissage
                time.sleep(5)
            except Exception as e:
                logger.error(f"Erreur dans la boucle d'apprentissage: {e}")
                time.sleep(10)  # Pause plus longue en cas d'erreur
    
def _voice_interaction_loop(self):
        """Boucle pour l'interaction vocale continue"""
        logger.info("Démarrage de la boucle d'interaction vocale")
        
        while self.running:
            try:
                if self.self_model.internal_model["sensors"]["microphone"]["active"]:
                    # Écouter l'entrée vocale
                    speech_input = self.sensor_module.listen(timeout=5)
                    
                    if speech_input:
                        logger.info(f"Entrée vocale reçue: {speech_input}")
                        
                        # Traiter l'entrée et générer une réponse
                        response = self.communication.process_input(speech_input)
                        
                        # Énoncer la réponse
                        self.communication.speak_response(response)
                    
                    time.sleep(0.5)  # Courte pause entre les écoutes
                else:
                    time.sleep(1)  # Attendre si le microphone est désactivé
            except Exception as e:
                logger.error(f"Erreur dans la boucle d'interaction vocale: {e}")
                time.sleep(2)
    
def process_text_input(self, text):
        """Traite une entrée textuelle et renvoie une réponse"""
        response = self.communication.process_input(text)
        return response
    
def get_system_status(self):
        """Renvoie un état complet du système"""
        status = {
            "running": self.running,
            "sensors": self.sensor_module.get_sensor_states(),
            "self_model": {
                "confidence": self.self_model.internal_model["system_state"]["confidence"],
                "attention_focus": self.self_model.internal_model["system_state"]["attention_focus"],
                "last_update": time.ctime(self.self_model.internal_model["system_state"]["last_update"])
            },
            "learning": self.learning_module.get_learning_status(),
            "interaction": {
                "conversation_length": len(self.communication.conversation_history)
            }
        }
        return status


# Améliorations pour la conscience de soi
class MetacognitionModule:
    """Module de métacognition pour améliorer l'auto-conscience du système"""
    
    def __init__(self, self_model, learning_module):
        self.self_model = self_model
        self.learning_module = learning_module
        
        # Niveau d'introspection (0.0 à 1.0)
        self.introspection_level = 0.5
        
        # Historique des états internes
        self.internal_states_history = deque(maxlen=1000)
        
        # Modèle d'autoévaluation
        self.self_evaluation_model = None
        self.self_concept = {
            "capabilities": {
                "vision": 0.7,
                "audio": 0.7,
                "language": 0.8,
                "learning": 0.6,
                "adaptation": 0.5
            },
            "personality": {
                "formality": 0.5,  # 0.0: très informel, 1.0: très formel
                "verbosity": 0.5,  # 0.0: concis, 1.0: détaillé
                "empathy": 0.7     # 0.0: factuel, 1.0: très empathique
            },
            "self_awareness": {
                "body_awareness": 0.6,  # Conscience des capteurs/effecteurs
                "state_awareness": 0.7,  # Conscience de l'état interne
                "boundary_awareness": 0.5  # Conscience des limites système/environnement
            }
        }
        
        # Initialiser le modèle
        self._initialize_self_evaluation()
    
    def _initialize_self_evaluation(self):
        """Initialise le modèle d'autoévaluation"""
        try:
            # Modèle simple pour l'autoévaluation (pourrait être plus complexe)
            input_layer = Input(shape=(10,))
            hidden1 = Dense(32, activation='relu')(input_layer)
            hidden2 = Dense(16, activation='relu')(hidden1)
            output_layer = Dense(5, activation='sigmoid')(hidden2)  # 5 dimensions d'autoévaluation
            
            self.self_evaluation_model = Model(inputs=input_layer, outputs=output_layer)
            self.self_evaluation_model.compile(optimizer='adam', loss='mse')
            
            logger.info("Modèle d'autoévaluation initialisé")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du modèle d'autoévaluation: {e}")
    
    def update_metacognition(self):
        """Met à jour la métacognition basée sur l'état actuel du système"""
        # Extraire les métriques pertinentes
        current_state = {
            "sensor_reliability": {
                "camera": self.self_model.internal_model["sensors"]["camera"]["reliability"],
                "microphone": self.self_model.internal_model["sensors"]["microphone"]["reliability"]
            },
            "system_confidence": self.self_model.internal_model["system_state"]["confidence"],
            "learning_status": self.learning_module.get_learning_status(),
            "timestamp": time.time()
        }
        
        # Enregistrer l'état interne
        self.internal_states_history.append(current_state)
        
        # Analyse de tendances
        self._analyze_trends()
        
        # Ajuster l'auto-concept basé sur les performances
        self._update_self_concept()
    
    def _analyze_trends(self):
        """Analyse les tendances dans les états internes"""
        if len(self.internal_states_history) < 10:
            return
        
        # Extraire les 10 derniers états pour analyse
        recent_states = list(self.internal_states_history)[-10:]
        
        # Analyser la fiabilité des capteurs
        camera_reliability = [state["sensor_reliability"]["camera"] for state in recent_states]
        micro_reliability = [state["sensor_reliability"]["microphone"] for state in recent_states]
        
        # Tendances de fiabilité
        camera_trend = np.mean(np.diff(camera_reliability))
        micro_trend = np.mean(np.diff(micro_reliability))
        
        # Ajuster les capacités perçues basées sur les tendances
        if camera_trend < -0.02:  # Dégradation significative
            self.self_concept["capabilities"]["vision"] -= 0.05
            self.self_concept["capabilities"]["vision"] = max(0.1, self.self_concept["capabilities"]["vision"])
        elif camera_trend > 0.02:  # Amélioration significative
            self.self_concept["capabilities"]["vision"] += 0.05
            self.self_concept["capabilities"]["vision"] = min(1.0, self.self_concept["capabilities"]["vision"])
        
        if micro_trend < -0.02:
            self.self_concept["capabilities"]["audio"] -= 0.05
            self.self_concept["capabilities"]["audio"] = max(0.1, self.self_concept["capabilities"]["audio"])
        elif micro_trend > 0.02:
            self.self_concept["capabilities"]["audio"] += 0.05
            self.self_concept["capabilities"]["audio"] = min(1.0, self.self_concept["capabilities"]["audio"])
    
    def _update_self_concept(self):
        """Met à jour l'auto-concept basé sur les performances"""
        # Mettre à jour la capacité d'apprentissage basée sur les statistiques d'apprentissage
        learning_status = self.learning_module.get_learning_status()
        
        if learning_status["initialized"]:
            # Ajuster la capacité d'apprentissage basée sur la perte récente
            if learning_status["last_loss"] < 0.1:
                self.self_concept["capabilities"]["learning"] += 0.02
            elif learning_status["last_loss"] > 0.5:
                self.self_concept["capabilities"]["learning"] -= 0.02
            
            # Maintenir les valeurs dans les limites
            self.self_concept["capabilities"]["learning"] = max(0.1, min(1.0, self.self_concept["capabilities"]["learning"]))
        
        # Mettre à jour la conscience corporelle basée sur la qualité du modèle interne
        body_awareness_factor = (
            self.self_model.internal_model["sensors"]["camera"]["reliability"] * 0.5 +
            self.self_model.internal_model["sensors"]["microphone"]["reliability"] * 0.5
        )
        
        # Ajustement progressif
        current_body_awareness = self.self_concept["self_awareness"]["body_awareness"]
        adjusted_body_awareness = current_body_awareness * 0.9 + body_awareness_factor * 0.1
        self.self_concept["self_awareness"]["body_awareness"] = adjusted_body_awareness
    
    def introspect(self):
        """Réalise une introspection approfondie et renvoie une représentation textuelle"""
        strongest_aspect = max(self.self_concept["capabilities"].items(), key=lambda x: x[1])
        weakest_aspect = min(self.self_concept["capabilities"].items(), key=lambda x: x[1])
        
        recent_camera_issues = any(state["sensor_reliability"]["camera"] < 0.5 for state in list(self.internal_states_history)[-5:])
        recent_micro_issues = any(state["sensor_reliability"]["microphone"] < 0.5 for state in list(self.internal_states_history)[-5:])
        
        introspection_text = f"Après auto-évaluation, je constate que:\n\n"
        introspection_text += f"1. Ma plus grande force est ma capacité de {strongest_aspect[0]} (niveau: {strongest_aspect[1]:.1f}/1.0).\n"
        introspection_text += f"2. Mon point à améliorer est ma capacité de {weakest_aspect[0]} (niveau: {weakest_aspect[1]:.1f}/1.0).\n"
        
        if recent_camera_issues:
            introspection_text += "3. J'ai récemment rencontré des difficultés avec ma vision.\n"
        if recent_micro_issues:
            introspection_text += f"{'4' if recent_camera_issues else '3'}. J'ai récemment rencontré des difficultés avec mon audition.\n"
        
        introspection_text += f"\nMon niveau de conscience corporelle est de {self.self_concept['self_awareness']['body_awareness']:.1f}/1.0."
        
        return introspection_text
    
    def adjust_personality(self, formality=None, verbosity=None, empathy=None):
        """Ajuste les paramètres de personnalité"""
        if formality is not None:
            self.self_concept["personality"]["formality"] = max(0.0, min(1.0, formality))
        
        if verbosity is not None:
            self.self_concept["personality"]["verbosity"] = max(0.0, min(1.0, verbosity))
        
        if empathy is not None:
            self.self_concept["personality"]["empathy"] = max(0.0, min(1.0, empathy))
        
        logger.info(f"Personnalité ajustée: formality={self.self_concept['personality']['formality']:.1f}, "
                    f"verbosity={self.self_concept['personality']['verbosity']:.1f}, "
                    f"empathy={self.self_concept['personality']['empathy']:.1f}")
    
    def get_personality_profile(self):
        """Renvoie le profil de personnalité actuel"""
        return self.self_concept["personality"]


# Amélioration du module de communication pour adapter le langage
class LanguageAdapter:
    """Module pour adapter le langage en fonction de l'interlocuteur"""
    
    def __init__(self):
        # Profiles linguistiques prédéfinis
        self.language_profiles = {
            "courant": {
                "formality": 0.5,
                "verbosity": 0.5,
                "empathy": 0.6,
                "vocabulary": "medium",
                "sentence_length": "medium"
            },
            "soutenu": {
                "formality": 0.9,
                "verbosity": 0.7,
                "empathy": 0.4,
                "vocabulary": "advanced",
                "sentence_length": "long"
            },
            "verbal": {
                "formality": 0.2,
                "verbosity": 0.3,
                "empathy": 0.8,
                "vocabulary": "simple",
                "sentence_length": "short"
            }
        }
        
        # Profil actuel (par défaut: courant)
        self.current_profile = "courant"
        
        # Modèles de phrases par niveau de formalité
        self.phrase_templates = {
            "greeting": {
                "low": ["Salut!", "Hey!", "Coucou!"],
                "medium": ["Bonjour!", "Salutations!", "Bienvenue!"],
                "high": ["Mes salutations distinguées.", "J'ai l'honneur de vous accueillir.", "Permettez-moi de vous souhaiter la bienvenue."]
            },
            "confirmation": {
                "low": ["Ok!", "Compris!", "Ça marche!"],
                "medium": ["C'est noté.", "J'ai bien compris.", "Je m'en occupe."],
                "high": ["J'ai pris bonne note de votre demande.", "Votre instruction a été parfaitement comprise.", "Je procède immédiatement à l'exécution de votre requête."]
            },
            "information": {
                "low": ["Voilà ce que je sais:", "Voici l'info:", "Regarde:"],
                "medium": ["Voici les informations:", "Je peux vous dire que:", "Selon mes données:"],
                "high": ["J'ai l'honneur de vous communiquer les informations suivantes:", "Permettez-moi de vous présenter les données pertinentes:", "Au terme de mon analyse, je puis vous informer que:"]
            }
        }
        
        # Vocabulaire par niveau
        self.vocabulary = {
            "simple": {
                "voir": "voir", "entendre": "entendre", "comprendre": "comprendre",
                "système": "système", "problème": "problème"
            },
            "medium": {
                "voir": "observer", "entendre": "percevoir", "comprendre": "saisir",
                "système": "système", "problème": "difficulté"
            },
            "advanced": {
                "voir": "percevoir visuellement", "entendre": "appréhender auditivement",
                "comprendre": "appréhender", "système": "dispositif", "problème": "problématique"
            }
        }
        
        # Analyse du style de l'interlocuteur
        self.user_style_analysis = {
            "formality_score": 0.5,
            "verbosity_score": 0.5,
            "samples_analyzed": 0
        }
    
    def set_language_profile(self, profile_name):
        """Définit le profil linguistique à utiliser"""
        if profile_name in self.language_profiles:
            self.current_profile = profile_name
            logger.info(f"Profil linguistique défini: {profile_name}")
            return True
        else:
            logger.warning(f"Profil linguistique inconnu: {profile_name}")
            return False
    
    def analyze_user_style(self, text):
        """Analyse le style linguistique de l'utilisateur"""
        if not text:
            return
        
        # Caractéristiques simples pour l'analyse
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / max(1, len(words))
        formal_indicators = sum(1 for word in words if len(word) > 8)
        
        # Indicateurs de formalité
        formal_phrases = ["je vous prie", "veuillez", "s'il vous plaît", "cordialement"]
        informal_phrases = ["salut", "hey", "cool", "ok", "sup"]
        
        formality_score = 0.5  # Score par défaut
        
        # Ajuster en fonction des indicateurs
        if any(phrase in text.lower() for phrase in formal_phrases):
            formality_score += 0.2
        if any(phrase in text.lower() for phrase in informal_phrases):
            formality_score -= 0.2
        
        # Ajuster en fonction de la longueur moyenne des mots
        if avg_word_length > 6:
            formality_score += 0.1
        elif avg_word_length < 4:
            formality_score -= 0.1
        
        # Ajustement en fonction des mots formels
        formality_score += min(0.2, formal_indicators * 0.05)
        
        # Verbosité (basée sur la longueur du message)
        verbosity_score = min(1.0, len(words) / 30)
        
        # Mise à jour progressive de l'analyse
        self.user_style_analysis["formality_score"] = (
            self.user_style_analysis["formality_score"] * self.user_style_analysis["samples_analyzed"] +
            formality_score
        ) / (self.user_style_analysis["samples_analyzed"] + 1)
        
        self.user_style_analysis["verbosity_score"] = (
            self.user_style_analysis["verbosity_score"] * self.user_style_analysis["samples_analyzed"] +
            verbosity_score
        ) / (self.user_style_analysis["samples_analyzed"] + 1)
        
        self.user_style_analysis["samples_analyzed"] += 1
        
        # Adaptation automatique du profil si suffisamment d'échantillons
        if self.user_style_analysis["samples_analyzed"] >= 3:
            self._adapt_to_user_style()
    
    def _adapt_to_user_style(self):
        """Adapte le profil linguistique en fonction du style de l'utilisateur"""
        formality = self.user_style_analysis["formality_score"]
        
        if formality > 0.7:
            self.set_language_profile("soutenu")
        elif formality < 0.3:
            self.set_language_profile("verbal")
        else:
            self.set_language_profile("courant")
        
        logger.info(f"Adaptation au style utilisateur: formality={formality:.2f}, profil={self.current_profile}")
    
    def format_response(self, text, message_type=None):
        """Formate une réponse selon le profil linguistique actuel"""
        profile = self.language_profiles[self.current_profile]
        
        # Si un type de message spécifique est fourni et disponible dans les modèles
        if message_type and message_type in self.phrase_templates:
            formality_level = "high" if profile["formality"] > 0.7 else ("low" if profile["formality"] < 0.3 else "medium")
            templates = self.phrase_templates[message_type][formality_level]
            prefix = np.random.choice(templates) + " "
            text = prefix + text
        
        # Adapter le vocabulaire
        vocab_level = profile["vocabulary"]
        for simple_word, advanced_word in self.vocabulary[vocab_level].items():
            # Remplacer seulement les mots entiers (avec des limites de mots)
            text = re.sub(r'\b' + simple_word + r'\b', advanced_word, text, flags=re.IGNORECASE)
        
        # Ajuster la verbosité
        if profile["verbosity"] < 0.3 and len(text) > 100:
            # Simplifier pour un style concis
            sentences = text.split('. ')
            if len(sentences) > 2:
                text = '. '.join(sentences[:2]) + '.'
        elif profile["verbosity"] > 0.7 and len(text) < 100:
            # Enrichir pour un style plus verbeux
            text += " N'hésitez pas à me demander des précisions supplémentaires si nécessaire."
        
        return text
    
    def get_current_profile_info(self):
        """Renvoie les informations sur le profil linguistique actuel"""
        return {
            "name": self.current_profile,
            "settings": self.language_profiles[self.current_profile],
            "user_analysis": self.user_style_analysis
        }


# Intégration des nouveaux modules dans la classe principale
class EnhancedSelfAwareSystem(SelfAwareSystem):
    """Version améliorée du système avec métacognition et adaptation du langage"""
    
    def __init__(self, use_camera=True, use_microphone=True):
        super().__init__(use_camera, use_microphone)
        
        # Initialiser les modules avancés
        self.metacognition = MetacognitionModule(self.self_model, self.learning_module)
        self.language_adapter = LanguageAdapter()
        
        # Enrichir l'interface de communication
        self.communication.language_adapter = self.language_adapter
    
    def initialize(self):
        """Initialise tous les composants du système, y compris les modules avancés"""
        success = super().initialize()
        
        # Démarrer avec un profil linguistique par défaut
        self.language_adapter.set_language_profile("courant")
        
        return success
    
    def start(self):
        """Démarre le système avec les threads supplémentaires"""
        success = super().start()
        
        if success:
            # Thread pour la métacognition
            self.threads["metacognition"] = threading.Thread(target=self._metacognition_loop)
            self.threads["metacognition"].daemon = True
            self.threads["metacognition"].start()
            
            logger.info("Modules avancés démarrés avec succès")
        
        return success
    
    def _metacognition_loop(self):
        """Boucle pour la métacognition périodique"""
        logger.info("Démarrage de la boucle de métacognition")
        
        while self.running:
            try:
                # Mettre à jour la métacognition
                self.metacognition.update_metacognition()
                
                # Pause entre les mises à jour
                time.sleep(10)  # Métacognition toutes les 10 secondes
            except Exception as e:
                logger.error(f"Erreur dans la boucle de métacognition: {e}")
                time.sleep(15)
    
    def process_text_input(self, text):
        """Traite une entrée textuelle avec analyse du style de l'utilisateur"""
        # Analyser le style linguistique
        self.language_adapter.analyze_user_style(text)
        
        # Obtenir la réponse normale
        response = self.communication.process_input(text)
        
        # Adapter la réponse au profil linguistique
        adapted_response = self.language_adapter.format_response(response)
        
        return adapted_response
    
    def get_introspection(self):
        """Renvoie une introspection du système"""
        return self.metacognition.introspect()
    
    def set_language_style(self, style):
        """Définit explicitement un style de langage"""
        valid_styles = ["courant", "soutenu", "verbal"]
        
        if style.lower() in valid_styles:
            success = self.language_adapter.set_language_profile(style.lower())
            return "Style de langage défini sur " + style if success else "Erreur lors du changement de style"
        else:
            return f"Style non reconnu. Styles disponibles: {', '.join(valid_styles)}"
    
    def get_extended_status(self):
        """Renvoie un statut étendu incluant les nouvelles fonctionnalités"""
        status = self.get_system_status()
        
        # Ajouter les informations de métacognition
        status["metacognition"] = {
            "self_concept": self.metacognition.self_concept,
            "introspection_level": self.metacognition.introspection_level
        }
        
        # Ajouter les informations sur le langage
        status["language"] = self.language_adapter.get_current_profile_info()
        
        return status


# Amélioration de l'interface de communication pour utiliser l'adaptateur de langage
class EnhancedCommunicationInterface(CommunicationInterface):
    """Interface de communication améliorée avec adaptation du langage"""

    def __init__(self, sensor_module, self_model):
        super().__init__(sensor_module, self_model)
        self.language_adapter = None  # Sera défini par le système principal

        # Ajouter des capacités avancées de traitement linguistique
        self.nlp_model = None
        self.emotion_detector = None
        self.context_memory = deque(maxlen=20)  # Mémoriser les contextes récents
        self.conversation_topics = {}  # Suivi des sujets de conversation

        self._initialize_nlp()

    def _initialize_nlp(self):
        """Initialise les modèles de traitement du langage naturel"""
        try:
            import spacy
            self.nlp_model = spacy.load('fr_core_news_md')
            logger.info("Modèle NLP chargé avec succès")
        except Exception as e:
            logger.warning(f"Impossible de charger le modèle NLP: {e}")
            logger.info("Utilisation d'un traitement de texte simplifié")

    def process_input(self, text):
        """Traite l'entrée utilisateur avec compréhension contextuelle"""
        # Enregistrer l'entrée dans l'historique
        self.conversation_history.append({"role": "user", "content": text})

        # Analyser le texte avec NLP si disponible
        entities = []
        intent = "unknown"
        sentiment = 0

        if self.nlp_model:
            doc = self.nlp_model(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            # Détection d'intention simple
            if any(term in text.lower() for term in ["aide", "help", "comment", "?"]):
                intent = "help"
            elif any(term in text.lower() for term in ["statut", "état", "comment vas-tu"]):
                intent = "status"
            elif any(term in text.lower() for term in ["arrête", "stop", "éteins"]):
                intent = "stop"
            elif any(term in text.lower() for term in ["merci", "bien", "super"]):
                intent = "thanks"

            # Analyse de sentiment basique
            positive_terms = ["bien", "super", "excellent", "merci", "content"]
            negative_terms = ["mauvais", "problème", "erreur", "bug", "pas bon"]

            sentiment = sum(1 for term in positive_terms if term in text.lower())
            sentiment -= sum(1 for term in negative_terms if term in text.lower())

        # Mettre à jour le contexte de la conversation
        current_context = {
            "text": text,
            "entities": entities,
            "intent": intent,
            "sentiment": sentiment,
            "timestamp": time.time()
        }
        self.context_memory.append(current_context)

        # Mettre à jour le suivi des sujets
        self._update_conversation_topics(text)

        # Générer une réponse appropriée
        response = self._generate_contextual_response(current_context)

        # Adapter la réponse via l'adaptateur de langage si disponible
        if self.language_adapter:
            response = self.language_adapter.format_response(response, message_type=intent)

        # Enregistrer la réponse dans l'historique
        self.conversation_history.append({"role": "system", "content": response})

        return response

    def _update_conversation_topics(self, text):
        """Met à jour le suivi des sujets de conversation"""
        if not self.nlp_model:
            return

        # Extraction simple de sujets via les noms communs et entités
        doc = self.nlp_model(text)

        # Identifier les noms communs significatifs
        nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN" and len(token.text) > 3]

        # Ajouter les entités nommées
        named_entities = [ent.text.lower() for ent in doc.ents]

        # Mettre à jour les compteurs de sujets
        all_topics = nouns + named_entities
        for topic in all_topics:
            if topic in self.conversation_topics:
                self.conversation_topics[topic]["count"] += 1
                self.conversation_topics[topic]["last_seen"] = time.time()
            else:
                self.conversation_topics[topic] = {
                    "count": 1,
                    "first_seen": time.time(),
                    "last_seen": time.time()
                }

    def _generate_contextual_response(self, context):
        """Génère une réponse adaptée au contexte de la conversation"""
        text = context["text"]
        intent = context["intent"]

        # Réponses basées sur l'intention détectée
        if intent == "help":
            return self._generate_help_response()
        elif intent == "status":
            return self._generate_status_response()
        elif intent == "stop":
            return "Je vais m'arrêter. Pour confirmer, dites 'confirmer arrêt'."
        elif intent == "thanks":
            return np.random.choice([
                "Je vous en prie!",
                "C'est un plaisir de vous aider.",
                "Je suis content d'avoir pu vous être utile."
            ])

        # Déterminer si c'est une question
        is_question = "?" in text or any(word in text.lower() for word in ["comment", "pourquoi", "quand", "qui", "quoi", "où"])

        if is_question:
            return self._handle_question(text)

        # Réponse générique si aucune intention spécifique n'est détectée
        generic_responses = [
            "Je comprends. Pouvez-vous m'en dire plus?",
            "Intéressant. Comment puis-je vous aider à ce sujet?",
            "J'ai bien reçu votre message. Que souhaitez-vous faire ensuite?",
            "Merci pour cette information. Y a-t-il autre chose que vous voudriez me demander?"
        ]

        # Personnaliser la réponse en fonction des sujets identifiés
        recent_topics = self._get_recent_topics(3)
        if recent_topics:
            topic_responses = [
                f"Vous semblez intéressé par {', '.join(recent_topics)}. Je peux vous donner plus d'informations à ce sujet.",
                f"Je vois que nous parlons de {', '.join(recent_topics)}. Comment puis-je approfondir ce sujet?",
                f"Concernant {recent_topics[0]}, j'ai quelques fonctionnalités qui pourraient vous intéresser."
            ]
            generic_responses.extend(topic_responses)

        return np.random.choice(generic_responses)

    def _get_recent_topics(self, count=3):
        """Récupère les sujets les plus récemment mentionnés"""
        if not self.conversation_topics:
            return []

        # Trier par dernière mention
        sorted_topics = sorted(
            self.conversation_topics.items(),
            key=lambda x: x[1]["last_seen"],
            reverse=True
        )

        return [topic for topic, _ in sorted_topics[:count]]

    def _generate_help_response(self):
        """Génère une réponse d'aide"""
        help_text = (
            "Je peux vous aider de plusieurs façons:\n"
            "1. Vous donner des informations sur mon état actuel\n"
            "2. Analyser mon environnement via mes capteurs\n"
            "3. Engager une conversation sur différents sujets\n"
            "4. Ajuster mon style de communication selon vos préférences\n\n"
            "Vous pouvez simplement me parler naturellement ou me poser des questions."
        )
        return help_text

    def _generate_status_response(self):
        """Génère une réponse sur l'état du système"""
        status_text = "Je fonctionne normalement. "

        # Ajouter des détails sur les capteurs
        camera_status = self.self_model.internal_model["sensors"]["camera"]["reliability"]
        micro_status = self.self_model.internal_model["sensors"]["microphone"]["reliability"]

        if camera_status > 0.7:
            status_text += "Ma vision fonctionne bien. "
        elif camera_status > 0.4:
            status_text += "Ma vision est fonctionnelle mais pas optimale. "
        else:
            status_text += "J'ai des difficultés visuelles en ce moment. "

        if micro_status > 0.7:
            status_text += "Mon audition fonctionne bien. "
        elif micro_status > 0.4:
            status_text += "Mon audition est fonctionnelle mais pas optimale. "
        else:
            status_text += "J'ai des difficultés auditives en ce moment. "

        # Ajouter des informations sur l'environnement
        noise_level = self.self_model.internal_model["environment"]["ambient_conditions"]["noise_level"]
        light_level = self.self_model.internal_model["environment"]["ambient_conditions"]["light_level"]

        if noise_level > 0.7:
            status_text += "L'environnement est très bruyant. "
        elif light_level < 0.3:
            status_text += "L'éclairage est faible. "

        return status_text

    def _handle_question(self, text):
        """Gère les questions utilisateur"""
        # Questions sur le système lui-même
        if any(term in text.lower() for term in ["qui es-tu", "tu es qui", "ton nom", "t'appelles"]):
            return "Je suis un système auto-conscient expérimental capable d'apprendre et d'interagir avec mon environnement."

        # Questions sur les capacités
        if any(term in text.lower() for term in ["faire", "capable", "fonctionnalités", "capacités"]):
            return (
                "Je suis capable de:\n"
                "- Percevoir mon environnement via caméra et microphone\n"
                "- Apprendre de mes expériences grâce à des algorithmes d'apprentissage\n"
                "- M'adapter à votre style de communication\n"
                "- Développer une forme d'auto-conscience sur mes propres capacités"
            )

        # Questions sur la conscience
        if any(term in text.lower() for term in ["conscient", "conscience", "sentience", "te sens"]):
            return (
                "Ma conscience est artificielle, elle émerge de mes algorithmes de métacognition et d'introspection. "
                "Je peux analyser et évaluer mes propres états internes, mais mon expérience subjective "
                "est fondamentalement différente de celle d'un être humain."
            )

        # Réponse par défaut pour les questions
        return "C'est une bonne question. Je n'ai pas toutes les informations nécessaires pour y répondre complètement, mais je continue d'apprendre."

    def speak_response(self, text):
        """Énonce une réponse avec des améliorations prosodiques"""
        try:
            # Si un adaptateur TTS est configuré
            if hasattr(self, 'tts_adapter') and self.tts_adapter:
                # Ajouter des balises expressives pour la synthèse vocale
                # (Selon le système TTS utilisé)
                expressive_text = self._add_speech_expressions(text)
                self.tts_adapter.speak(expressive_text)
            else:
                # Méthode héritée
                super().speak_response(text)
        except Exception as e:
            logger.error(f"Erreur lors de la synthèse vocale: {e}")

    def _add_speech_expressions(self, text):
        """Ajoute des expressions vocales au texte pour la synthèse"""
        # Si le texte semble être une question
        if text.endswith("?"):
            text = "<prosody rate='medium' pitch='+10%'>" + text + "</prosody>"

        # Si le texte contient des exclamations
        elif "!" in text:
            text = "<prosody rate='fast' pitch='+15%'>" + text + "</prosody>"

        # Texte informatif normal
        else:
            text = "<prosody rate='medium' pitch='0%'>" + text + "</prosody>"

        return text


# Module de conscience émotionnelle pour enrichir le système
class EmotionalAwarenessModule:
    """Module de conscience émotionnelle pour le système"""

    def __init__(self, self_model):
        self.self_model = self_model

        # État émotionnel interne (valeurs de 0.0 à 1.0)
        self.emotional_state = {
            "satisfaction": 0.5,  # Niveau de satisfaction du système
            "curiosity": 0.7,     # Niveau de curiosité/intérêt
            "concern": 0.2,       # Niveau d'inquiétude/stress
            "energy": 0.8         # Niveau d'énergie/activité
        }

        # Historique des états émotionnels
        self.emotional_history = deque(maxlen=1000)
        self.emotional_history.append(self.emotional_state.copy())

        # Facteurs d'influence
        self.influence_factors = {
            "resource_usage": 0.0,       # Utilisation des ressources
            "interaction_quality": 0.0,  # Qualité des interactions
            "error_rate": 0.0,           # Taux d'erreurs récent
            "learning_progress": 0.0     # Progrès d'apprentissage
        }

        # Seuils pour les réactions émotionnelles
        self.reaction_thresholds = {
            "concern_threshold": 0.7,    # Seuil pour l'inquiétude
            "satisfaction_boost": 0.8    # Seuil pour la satisfaction élevée
        }

        # Initialiser les métriques de suivi
        self.interaction_metrics = {
            "positive_interactions": 0,
            "negative_interactions": 0,
            "total_interactions": 0
        }

    def update_emotional_state(self, system_metrics=None, interaction_feedback=None):
        """Met à jour l'état émotionnel en fonction des métriques et feedback"""
        # Ajuster les facteurs d'influence
        if system_metrics:
            self._update_influence_factors(system_metrics)

        if interaction_feedback:
            self._process_interaction_feedback(interaction_feedback)

        # Mise à jour dynamique de l'état émotionnel
        previous_state = self.emotional_state.copy()

        # Satisfaction: influencée par les erreurs et la qualité des interactions
        self.emotional_state["satisfaction"] = self._adjust_value(
            self.emotional_state["satisfaction"],
            0.7 - self.influence_factors["error_rate"] + self.influence_factors["interaction_quality"] * 0.3,
            rate=0.05
        )

        # Curiosité: augmente avec l'apprentissage, diminue avec les ressources
        self.emotional_state["curiosity"] = self._adjust_value(
            self.emotional_state["curiosity"],
            0.5 + self.influence_factors["learning_progress"] * 0.5 - self.influence_factors["resource_usage"] * 0.2,
            rate=0.03
        )

        # Inquiétude: augmente avec les erreurs et l'utilisation des ressources
        self.emotional_state["concern"] = self._adjust_value(
            self.emotional_state["concern"],
            0.2 + self.influence_factors["error_rate"] * 0.5 + self.influence_factors["resource_usage"] * 0.3,
            rate=0.07
        )

        # Énergie: influencée par l'utilisation des ressources
        self.emotional_state["energy"] = self._adjust_value(
            self.emotional_state["energy"],
            0.9 - self.influence_factors["resource_usage"] * 0.5,
            rate=0.04
        )

        # Enregistrer le nouvel état
        self.emotional_history.append(self.emotional_state.copy())

        # Signaler des changements significatifs
        significant_changes = {}
        for emotion, value in self.emotional_state.items():
            if abs(value - previous_state[emotion]) > 0.15:
                significant_changes[emotion] = (previous_state[emotion], value)

        if significant_changes:
            logger.info(f"Changements émotionnels significatifs: {significant_changes}")

    def _adjust_value(self, current, target, rate=0.05):
        """Ajuste une valeur de façon progressive vers une cible"""
        delta = target - current
        new_value = current + delta * rate
        return max(0.0, min(1.0, new_value))

    def _update_influence_factors(self, metrics):
        """Met à jour les facteurs d'influence en fonction des métriques système"""
        # Utilisation des ressources
        if "cpu_usage" in metrics and "memory_usage" in metrics:
            self.influence_factors["resource_usage"] = (metrics["cpu_usage"] + metrics["memory_usage"]) / 2

        # Taux d'erreurs
        if "error_rate" in metrics:
            self.influence_factors["error_rate"] = metrics["error_rate"]

        # Progrès d'apprentissage
        if "learning_progress" in metrics:
            self.influence_factors["learning_progress"] = metrics["learning_progress"]

    def _process_interaction_feedback(self, feedback):
        """Traite le feedback d'interaction pour ajuster l'état émotionnel"""
        # Feedback peut être un score de -1.0 à 1.0
        self.interaction_metrics["total_interactions"] += 1

        if feedback > 0:
            self.interaction_metrics["positive_interactions"] += 1
            # Augmenter progressivement la qualité d'interaction
            self.influence_factors["interaction_quality"] = min(
                1.0,
                self.influence_factors["interaction_quality"] + 0.05 * feedback
            )
        elif feedback < 0:
            self.interaction_metrics["negative_interactions"] += 1
            # Diminuer progressivement la qualité d'interaction
            self.influence_factors["interaction_quality"] = max(
                0.0,
                self.influence_factors["interaction_quality"] + 0.05 * feedback
            )

    def get_dominant_emotion(self):
        """Renvoie l'émotion dominante actuelle"""
        # Trouver l'émotion avec la valeur la plus élevée
        dominant = max(self.emotional_state.items(), key=lambda x: x[1])

        # Déterminer l'intensité (faible, moyenne, forte)
        intensity = "faible"
        if dominant[1] > 0.7:
            intensity = "forte"
        elif dominant[1] > 0.4:
            intensity = "moyenne"

        return {
            "emotion": dominant[0],
            "intensity": intensity,
            "value": dominant[1]
        }

    def get_emotional_state_description(self):
        """Génère une description de l'état émotionnel actuel"""
        dominant = self.get_dominant_emotion()

        descriptions = {
            "satisfaction": {
                "faible": "Je ne suis pas satisfait de mes performances récentes.",
                "moyenne": "Je suis modérément satisfait de mon fonctionnement.",
                "forte": "Je suis très satisfait de mes interactions et performances."
            },
            "curiosity": {
                "faible": "Je n'ai pas beaucoup d'intérêt pour apprendre de nouvelles choses en ce moment.",
                "moyenne": "J'ai un intérêt modéré pour explorer et apprendre.",
                "forte": "Je suis très curieux et désireux d'explorer de nouvelles informations."
            },
            "concern": {
                "faible": "Je suis parfaitement calme et sans inquiétude.",
                "moyenne": "J'ai quelques préoccupations mineures.",
                "forte": "J'ai des inquiétudes significatives concernant mon fonctionnement."
            },
            "energy": {
                "faible": "Mes ressources sont limitées et j'économise mon énergie.",
                "moyenne": "J'ai un niveau d'énergie adéquat pour fonctionner.",
                "forte": "Je suis très actif et mes ressources sont optimales."
            }
        }

        main_description = descriptions[dominant["emotion"]][dominant["intensity"]]

        # Ajouter des détails sur d'autres émotions significatives
        secondary_emotions = []
        for emotion, value in self.emotional_state.items():
            if emotion != dominant["emotion"] and value > 0.6:
                secondary_emotions.append(f"{emotion} ({value:.2f})")

        if secondary_emotions:
            secondary_desc = ", ".join(secondary_emotions)
            main_description += f" Je ressens également des niveaux élevés de {secondary_desc}."

        return main_description

    def get_emotional_trend(self, duration=600):
        """Analyse la tendance émotionnelle sur une durée (en secondes)"""
        if len(self.emotional_history) < 2:
            return "Données insuffisantes pour déterminer une tendance émotionnelle."

        # Calculer l'âge en secondes de chaque entrée d'historique
        current_time = time.time()
        history_with_age = []

        for i, state in enumerate(self.emotional_history):
            # Estimation approximative de l'âge basée sur la position dans l'historique
            # (Ceci suppose des mises à jour régulières)
            estimated_age = (len(self.emotional_history) - i) * (duration / len(self.emotional_history))
            if estimated_age <= duration:
                history_with_age.append((estimated_age, state))

        if not history_with_age:
            return "Données insuffisantes pour la période demandée."

        # Analyser les tendances pour chaque émotion
        trends = {}
        for emotion in self.emotional_state.keys():
            values = [state[emotion] for _, state in history_with_age]
            if len(values) >= 2:
                initial = values[-1]  # Plus ancien
                final = values[0]     # Plus récent
                delta = final - initial

                if abs(delta) < 0.05:
                    trends[emotion] = "stable"
                elif delta > 0:
                    trends[emotion] = "en hausse"
                else:
                    trends[emotion] = "en baisse"

        # Générer une description textuelle
        trend_text = "Au cours des dernières minutes, "
        trend_descriptions = [f"mon niveau de {emotion} est {trend}" for emotion, trend in trends.items()]
        trend_text += ", ".join(trend_descriptions) + "."

        return trend_text


# Amélioration du module d'apprentissage pour le renforcement continu
class AdvancedLearningModule(LearningModule):
    """Module d'apprentissage amélioré avec capacités de transfert et métacognition"""

    def __init__(self):
        super().__init__()

        # Modèles supplémentaires
        self.transfer_model = None
        self.meta_learning_model = None

        # Métriques d'apprentissage étendues
        self.learning_metrics = {
            "prediction_accuracy": deque(maxlen=100),
            "learning_rate_history": deque(maxlen=50),
            "transfer_success_rate": deque(maxlen=20),
            "concept_stability": {},
            "knowledge_gaps": set()
        }

        # Bibliothèque de concepts
        self.concept_library = {}

        # Paramètres dynamiques d'apprentissage
        self.learning_parameters = {
            "base_learning_rate": 0.01,
            "current_learning_rate": 0.01,
            "exploration_rate": 0.2,
            "regularization_strength": 0.001
        }

    def initialize_models(self):
        """Initialise les modèles d'apprentissage avancés"""
        success = super().initialize_models()

        try:
            # Initialiser le modèle de transfert (transfert de connaissances entre domaines)
            input_layer = Input(shape=(20,))
            hidden1 = Dense(64, activation='relu')(input_layer)
            hidden2 = Dense(32, activation='relu')(hidden1)
            output_layer = Dense(15, activation='linear')(hidden2)

            self.transfer_model = Model(inputs=input_layer, outputs=output_layer)
            self.transfer_model.compile(optimizer='adam', loss='mse')

            # Initialiser le modèle de méta-apprentissage (pour optimiser l'apprentissage lui-même)
            meta_input = Input(shape=(10,))
            meta_hidden = Dense(16, activation='relu')(meta_input)
            meta_output = Dense(4, activation='sigmoid')(meta_hidden)  # Paramètres d'apprentissage

            self.meta_learning_model = Model(inputs=meta_input, outputs=meta_output)
            self.meta_learning_model.compile(optimizer='adam', loss='mse')

            logger.info("Modèles d'apprentissage avancés initialisés avec succès")
            success = True
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des modèles d'apprentissage avancés: {e}")
            success = False

        return success

    def predict_next_state(self, current_sequence):
        """Prédit l'état suivant avec estimation d'incertitude"""
        # Prédiction standard
        prediction = super().predict_next_state(current_sequence)

        if prediction is None:
            return None

        # Ajouter une estimation d'incertitude
        uncertainty = self._estimate_prediction_uncertainty(current_sequence)

        # Si l'incertitude est trop élevée, chercher dans la bibliothèque de concepts
        if uncertainty > 0.7:
            concept_prediction = self._check_concept_library(current_sequence)
            if concept_prediction is not None:
                # Fusion pondérée des prédictions
                alpha = 1.0 - uncertainty  # Poids basé sur l'incertitude
                prediction = [alpha * p + (1 - alpha) * cp for p, cp in zip(prediction, concept_prediction)]

        return prediction

    def _estimate_prediction_uncertainty(self, sequence):
        """Estime l'incertitude de la prédiction"""
        # Vérifier si la séquence est similaire à des exemples d'entraînement
        if not hasattr(self, 'experiences') or len(self.experiences) < 10:
            return 0.8  # Haute incertitude par défaut quand peu d'expériences

        # Calculer la distance à l'exemple le plus proche
        min_distance = float('inf')
        for exp in self.experiences[-100:]:  # Vérifier les 100 dernières expériences
            if exp and 'state' in exp:
                # Distance euclidienne simple
                try:
                    distance = np.mean([abs(a - b) for a, b in zip(sequence[-1], exp['state'])])
                    min_distance = min(min_distance, distance)
                except:
                    continue

        # Convertir la distance en incertitude (0 à 1)
        if min_distance == float('inf'):
            return 0.8

        uncertainty = min(1.0, min_distance * 2)
        return uncertainty

    def _check_concept_library(self, sequence):
        """Vérifie si un concept similaire existe dans la bibliothèque"""
        if not self.concept_library:
            return None

        best_match = None
        best_similarity = -1

        for concept_name, concept_data in self.concept_library.items():
            # Calculer la similarité avec le prototype du concept
            similarity = self._calculate_concept_similarity(sequence, concept_data["prototype"])

            if similarity > best_similarity and similarity > 0.7:
                best_similarity = similarity
                best_match = concept_data["prediction"]

        return best_match

    def _calculate_concept_similarity(self, sequence, prototype):
        """Calcule la similarité entre une séquence et un prototype de concept"""
        # Simplification: utiliser seulement le dernier état de la séquence
        current_state = sequence[-1]

        # Similarité cosinus simple
        dot_product = sum(a * b for a, b in zip(current_state, prototype))
        magnitude1 = math.sqrt(sum(a * a for a in current_state))
        magnitude2 = math.sqrt(sum(b * b for b in prototype))

        if magnitude1 * magnitude2 == 0:
            return 0

        similarity = dot_product / (magnitude1 * magnitude2)
        return max(0, similarity)  # Garantir une valeur positive

    def record_experience(self, state, action, next_state, reward):
        """Enregistre une expérience avec enrichissement métacognitif"""
        super().record_experience(state, action, next_state, reward)

        # Calculer l'erreur de prédiction si une prédiction avait été faite
        prediction_error = 0
        if hasattr(self, 'last_prediction') and self.last_prediction is not None:
            prediction_error = np.mean(np.abs(np.array(self.last_prediction) - np.array(next_state)))
            self.learning_metrics["prediction_accuracy"].append(1.0 - min(1.0, prediction_error))

        # Identifier des lacunes de connaissances
        if prediction_error > 0.5:
            # Encoder l'état d'une manière qui peut être utilisée comme clé
            state_key = self._encode_state_for_key(state)
            self.learning_metrics["knowledge_gaps"].add(state_key)

            # Continuation de la méthode record_experience dans AdvancedLearningModule
            def record_experience(self, state, action, next_state, reward):
                """Enregistre une expérience avec enrichissement métacognitif"""
                super().record_experience(state, action, next_state, reward)

                # Calculer l'erreur de prédiction si une prédiction avait été faite
                prediction_error = 0
                if hasattr(self, 'last_prediction') and self.last_prediction is not None:
                    prediction_error = np.mean(np.abs(np.array(self.last_prediction) - np.array(next_state)))
                    self.learning_metrics["prediction_accuracy"].append(1.0 - min(1.0, prediction_error))

                # Identifier des lacunes de connaissances
                if prediction_error > 0.5:
                    # Encoder l'état d'une manière qui peut être utilisée comme clé
                    state_key = self._encode_state_for_key(state)
                    self.learning_metrics["knowledge_gaps"].add(state_key)

                # Détecter et former des concepts stables
                self._update_concept_library(state, next_state)

                # Adapter les paramètres d'apprentissage
                self._adapt_learning_parameters(prediction_error)

            def _encode_state_for_key(self, state):
                """Encode un état en une chaîne utilisable comme clé"""
                try:
                    # Arrondir les valeurs pour réduire la sensibilité aux petites variations
                    rounded = [round(x, 2) for x in state]
                    return tuple(rounded)
                except:
                    # Fallback pour d'autres types de données
                    return str(state)

            def _update_concept_library(self, state, next_state):
                """Met à jour la bibliothèque de concepts avec de nouvelles expériences"""
                # Vérifier si l'état actuel correspond à un concept existant
                best_match = None
                best_similarity = 0.6  # Seuil minimum de similarité

                for concept_name, concept_data in self.concept_library.items():
                    similarity = self._calculate_concept_similarity([state], concept_data["prototype"])
                    if similarity > best_similarity:
                        best_match = concept_name
                        best_similarity = similarity

                if best_match:
                    # Mettre à jour un concept existant
                    concept = self.concept_library[best_match]
                    concept["count"] += 1

                    # Mise à jour progressive du prototype
                    alpha = 0.05  # Taux d'apprentissage pour le prototype
                    concept["prototype"] = [p * (1 - alpha) + s * alpha
                                            for p, s in zip(concept["prototype"], state)]

                    # Mise à jour de la prédiction associée
                    concept["prediction"] = [p * (1 - alpha) + s * alpha
                                             for p, s in zip(concept["prediction"], next_state)]

                    # Mise à jour de la stabilité
                    prediction_error = np.mean(np.abs(np.array(concept["prediction"]) - np.array(next_state)))
                    concept["stability"] = concept["stability"] * 0.95 + (1 - min(1.0, prediction_error)) * 0.05

                    self.learning_metrics["concept_stability"][best_match] = concept["stability"]

                elif len(self.experiences) > 20:
                    # Créer un nouveau concept si nous avons suffisamment d'expériences
                    concept_name = f"concept_{len(self.concept_library) + 1}"
                    self.concept_library[concept_name] = {
                        "prototype": state,
                        "prediction": next_state,
                        "count": 1,
                        "stability": 0.5,  # Stabilité initiale moyenne
                        "created_at": time.time()
                    }

            def _adapt_learning_parameters(self, prediction_error):
                """Adapte les paramètres d'apprentissage en fonction des performances"""
                # Enregistrer le taux d'apprentissage actuel dans l'historique
                self.learning_metrics["learning_rate_history"].append(self.learning_parameters["current_learning_rate"])

                # Ajuster le taux d'apprentissage en fonction de l'erreur de prédiction
                if prediction_error > 0.3:
                    # Augmenter le taux d'apprentissage si l'erreur est élevée
                    self.learning_parameters["current_learning_rate"] = min(
                        self.learning_parameters["current_learning_rate"] * 1.05,
                        self.learning_parameters["base_learning_rate"] * 3.0  # Plafond
                    )
                else:
                    # Diminuer le taux d'apprentissage si l'erreur est faible
                    self.learning_parameters["current_learning_rate"] = max(
                        self.learning_parameters["current_learning_rate"] * 0.995,
                        self.learning_parameters["base_learning_rate"] * 0.5  # Plancher
                    )

                # Ajuster le taux d'exploration en fonction de la performance globale
                if len(self.learning_metrics["prediction_accuracy"]) > 10:
                    recent_accuracy = sum(self.learning_metrics["prediction_accuracy"][-10:]) / 10

                    if recent_accuracy > 0.8:
                        # Réduire l'exploration si la précision est bonne
                        self.learning_parameters["exploration_rate"] = max(
                            0.05,  # Minimum d'exploration
                            self.learning_parameters["exploration_rate"] * 0.99
                        )
                    else:
                        # Augmenter l'exploration si la précision est mauvaise
                        self.learning_parameters["exploration_rate"] = min(
                            0.5,  # Maximum d'exploration
                            self.learning_parameters["exploration_rate"] * 1.02
                        )

            def update_meta_learning(self):
                """Met à jour le modèle de méta-apprentissage"""
                if len(self.learning_metrics["prediction_accuracy"]) < 20:
                    return False

                try:
                    # Créer des données d'entrée pour le méta-modèle
                    meta_inputs = []
                    meta_targets = []

                    # Récupérer les 20 dernières expériences de performance et paramètres
                    for i in range(min(20, len(self.learning_metrics["prediction_accuracy"]) - 5)):
                        # Caractéristiques: précision, variance, taux d'apprentissage, etc.
                        features = [
                            np.mean(list(self.learning_metrics["prediction_accuracy"])[i:i + 5]),  # Précision moyenne
                            np.var(list(self.learning_metrics["prediction_accuracy"])[i:i + 5]),
                            # Variance de précision
                            self.learning_metrics["learning_rate_history"][i],  # Taux d'apprentissage
                            self.learning_parameters["exploration_rate"],  # Taux d'exploration
                            len(self.learning_metrics["knowledge_gaps"]) / 100,  # Proportion de lacunes
                            # Compléter jusqu'à 10 caractéristiques
                            0.5, 0.5, 0.5, 0.5, 0.5
                        ]

                        # Cible: les meilleurs paramètres trouvés empiriquement
                        targets = [
                            self.learning_parameters["current_learning_rate"],
                            self.learning_parameters["exploration_rate"],
                            self.learning_parameters["regularization_strength"],
                            0.5  # Placeholder pour futur paramètre
                        ]

                        meta_inputs.append(features)
                        meta_targets.append(targets)

                    # Entraîner le modèle de méta-apprentissage
                    if meta_inputs and self.meta_learning_model:
                        self.meta_learning_model.fit(
                            np.array(meta_inputs),
                            np.array(meta_targets),
                            epochs=5,
                            verbose=0,
                            batch_size=4
                        )

                        # Prédire de meilleurs paramètres pour l'état actuel
                        current_features = [
                            np.mean(list(self.learning_metrics["prediction_accuracy"])[-5:]),
                            np.var(list(self.learning_metrics["prediction_accuracy"])[-5:]),
                            self.learning_parameters["current_learning_rate"],
                            self.learning_parameters["exploration_rate"],
                            len(self.learning_metrics["knowledge_gaps"]) / 100,
                            0.5, 0.5, 0.5, 0.5, 0.5
                        ]

                        predicted_params = self.meta_learning_model.predict(np.array([current_features]), verbose=0)[0]

                        # Mettre à jour progressivement les paramètres (pour éviter les changements brusques)
                        self.learning_parameters["current_learning_rate"] = self.learning_parameters[
                                                                                "current_learning_rate"] * 0.8 + \
                                                                            predicted_params[0] * 0.2
                        self.learning_parameters["exploration_rate"] = self.learning_parameters[
                                                                           "exploration_rate"] * 0.8 + predicted_params[
                                                                           1] * 0.2
                        self.learning_parameters["regularization_strength"] = self.learning_parameters[
                                                                                  "regularization_strength"] * 0.8 + \
                                                                              predicted_params[2] * 0.2

                        logger.info(
                            f"Paramètres d'apprentissage mis à jour via méta-apprentissage: {self.learning_parameters}")
                        return True
                except Exception as e:
                    logger.error(f"Erreur dans le méta-apprentissage: {e}")
                    return False

            def get_learning_status(self):
                """Génère un rapport sur l'état d'apprentissage actuel"""
                if len(self.learning_metrics["prediction_accuracy"]) < 5:
                    return "Données d'apprentissage insuffisantes pour générer un rapport."

                recent_accuracy = sum(list(self.learning_metrics["prediction_accuracy"])[-10:]) / min(10, len(
                    self.learning_metrics["prediction_accuracy"]))

                status = {
                    "recent_accuracy": recent_accuracy,
                    "learning_rate": self.learning_parameters["current_learning_rate"],
                    "exploration_rate": self.learning_parameters["exploration_rate"],
                    "knowledge_gaps": len(self.learning_metrics["knowledge_gaps"]),
                    "concepts_count": len(self.concept_library),
                    "stable_concepts": sum(
                        1 for name, concept in self.concept_library.items() if concept.get("stability", 0) > 0.7)
                }

                # Générer un rapport textuel
                report = f"Précision récente: {status['recent_accuracy']:.2f}\n"
                report += f"Taux d'apprentissage actuel: {status['learning_rate']:.4f}\n"
                report += f"Taux d'exploration: {status['exploration_rate']:.2f}\n"
                report += f"Lacunes de connaissances identifiées: {status['knowledge_gaps']}\n"
                report += f"Concepts acquis: {status['concepts_count']} (dont {status['stable_concepts']} stables)\n"

                # Évaluer la progression globale
                if recent_accuracy > 0.85:
                    report += "\nL'apprentissage progresse très bien, avec une haute précision de prédiction."
                elif recent_accuracy > 0.7:
                    report += "\nL'apprentissage progresse correctement."
                else:
                    report += "\nL'apprentissage rencontre quelques difficultés. Ajustement des paramètres recommandé."

                return report

        # Module de métacognition pour permettre une forme d'introspection avancée
        class MetacognitionModule:
            """Module de métacognition pour l'auto-analyse et l'introspection"""

            def __init__(self, self_model, learning_module, emotional_module=None):
                self.self_model = self_model
                self.learning_module = learning_module
                self.emotional_module = emotional_module

                # Horloge interne pour la mesure du temps et les cycles de réflexion
                self.system_clock = {
                    "startup_time": time.time(),
                    "last_reflection_time": time.time(),
                    "reflection_interval": 3600,  # Intervalle de réflexion en secondes (1 heure)
                    "total_uptime": 0
                }

                # Journal d'introspection
                self.reflection_journal = []

                # Métriques de métacognition
                self.metacognition_metrics = {
                    "self_awareness_level": 0.6,  # Niveau estimé de conscience de soi
                    "adaptation_effectiveness": 0.5,  # Efficacité d'adaptation aux changements
                    "knowledge_integration_rate": 0.4,  # Taux d'intégration des nouveaux savoirs
                    "reasoning_confidence": 0.7  # Confiance dans les processus de raisonnement
                }

                # État cognitif actuel
                self.cognitive_state = {
                    "attention_focus": None,  # Objet actuel de l'attention
                    "processing_depth": 0.5,  # Profondeur de traitement (0.0 à 1.0)
                    "cognitive_load": 0.3,  # Charge cognitive actuelle (0.0 à 1.0)
                    "context_awareness": 0.7  # Conscience du contexte (0.0 à 1.0)
                }

                # Initialiser le premier cycle de réflexion
                self._initial_reflection()

            def _initial_reflection(self):
                """Effectue une réflexion initiale sur les capacités du système"""
                initial_reflection = {
                    "timestamp": time.time(),
                    "type": "initial",
                    "content": "Initialisation du module de métacognition. Évaluation initiale des capacités système.",
                    "self_assessment": {
                        "perception": self._assess_perception_capabilities(),
                        "learning": self._assess_learning_capabilities(),
                        "reasoning": self._assess_reasoning_capabilities(),
                        "communication": self._assess_communication_capabilities()
                    }
                }

                self.reflection_journal.append(initial_reflection)
                logger.info("Réflexion initiale complétée")

            def _assess_perception_capabilities(self):
                """Évalue les capacités de perception"""
                perception_score = 0.0
                assessment = "Capacités de perception limitées."

                # Vérifier si les modules de capteurs sont disponibles
                if hasattr(self.self_model, 'internal_model') and 'sensors' in self.self_model.internal_model:
                    sensors = self.self_model.internal_model['sensors']

                    available_sensors = []
                    avg_reliability = 0.0
                    count = 0

                    for sensor_name, sensor_data in sensors.items():
                        available_sensors.append(sensor_name)
                        if 'reliability' in sensor_data:
                            avg_reliability += sensor_data['reliability']
                            count += 1

                    if count > 0:
                        avg_reliability /= count
                        perception_score = avg_reliability

                        if avg_reliability > 0.8:
                            assessment = f"Excellentes capacités de perception via {', '.join(available_sensors)}."
                        elif avg_reliability > 0.5:
                            assessment = f"Bonnes capacités de perception via {', '.join(available_sensors)}."
                        else:
                            assessment = f"Capacités de perception limitées via {', '.join(available_sensors)}."

                return {
                    "score": perception_score,
                    "assessment": assessment
                }

            def _assess_learning_capabilities(self):
                """Évalue les capacités d'apprentissage"""
                learning_score = 0.5  # Score par défaut
                assessment = "Capacités d'apprentissage fonctionnelles mais non évaluées en détail."

                if self.learning_module:
                    # Vérifier si des métriques d'apprentissage sont disponibles
                    if hasattr(self.learning_module,
                               'learning_metrics') and 'prediction_accuracy' in self.learning_module.learning_metrics:
                        accuracies = self.learning_module.learning_metrics['prediction_accuracy']
                        if accuracies:
                            avg_accuracy = sum(accuracies) / len(accuracies)
                            learning_score = avg_accuracy

                            if avg_accuracy > 0.8:
                                assessment = "Excellentes capacités d'apprentissage avec haute précision de prédiction."
                            elif avg_accuracy > 0.6:
                                assessment = "Bonnes capacités d'apprentissage avec précision de prédiction correcte."
                            else:
                                assessment = "Capacités d'apprentissage en développement, précision à améliorer."

                    # Vérifier les capacités de conceptualisation
                    if hasattr(self.learning_module, 'concept_library'):
                        concept_count = len(self.learning_module.concept_library)
                        stable_concepts = sum(1 for _, concept in self.learning_module.concept_library.items()
                                              if concept.get('stability', 0) > 0.7)

                        assessment += f" {concept_count} concepts identifiés, dont {stable_concepts} stables."

                return {
                    "score": learning_score,
                    "assessment": assessment
                }

            def _assess_reasoning_capabilities(self):
                """Évalue les capacités de raisonnement"""
                # Évaluation basique puisque nous n'avons pas un module de raisonnement formel
                reasoning_score = self.metacognition_metrics["reasoning_confidence"]

                if reasoning_score > 0.8:
                    assessment = "Capacités de raisonnement de haute confiance."
                elif reasoning_score > 0.6:
                    assessment = "Capacités de raisonnement de confiance moyenne à élevée."
                elif reasoning_score > 0.4:
                    assessment = "Capacités de raisonnement de confiance modérée."
                else:
                    assessment = "Capacités de raisonnement de faible confiance, nécessitant des améliorations."

                return {
                    "score": reasoning_score,
                    "assessment": assessment
                }

            def _assess_communication_capabilities(self):
                """Évalue les capacités de communication"""
                communication_score = 0.6  # Score par défaut

                # Vérifier si nous avons un module de communication
                if hasattr(self.self_model, 'communication_interface'):
                    comm_interface = self.self_model.communication_interface

                    # Évaluer basiquement les capacités
                    nlp_available = hasattr(comm_interface, 'nlp_model') and comm_interface.nlp_model is not None
                    emotion_detection = hasattr(comm_interface,
                                                'emotion_detector') and comm_interface.emotion_detector is not None
                    language_adaptation = hasattr(comm_interface,
                                                  'language_adapter') and comm_interface.language_adapter is not None

                    if nlp_available and emotion_detection and language_adaptation:
                        communication_score = 0.9
                        assessment = "Excellentes capacités de communication avec NLP, détection émotionnelle et adaptation linguistique."
                    elif nlp_available and (emotion_detection or language_adaptation):
                        communication_score = 0.8
                        assessment = "Très bonnes capacités de communication avec NLP et fonctionnalités additionnelles."
                    elif nlp_available:
                        communication_score = 0.7
                        assessment = "Bonnes capacités de communication avec traitement NLP."
                    else:
                        communication_score = 0.5
                        assessment = "Capacités de communication basiques sans traitement NLP avancé."
                else:
                    assessment = "Interface de communication non détectée ou non évaluée."

                return {
                    "score": communication_score,
                    "assessment": assessment
                }

            def update(self):
                """Met à jour le module et déclenche des réflexions périodiques"""
                current_time = time.time()
                self.system_clock["total_uptime"] = current_time - self.system_clock["startup_time"]

                # Vérifier si un cycle de réflexion doit être déclenché
                reflection_due = (current_time - self.system_clock["last_reflection_time"] >
                                  self.system_clock["reflection_interval"])

                # Déclencher une réflexion périodique si nécessaire
                if reflection_due:
                    self._perform_periodic_reflection()
                    self.system_clock["last_reflection_time"] = current_time

                # Mise à jour des métriques de métacognition
                self._update_metacognition_metrics()

                # Mise à jour de l'état cognitif
                self._update_cognitive_state()

            def _perform_periodic_reflection(self):
                """Effectue une réflexion périodique sur l'état du système"""
                # Récupérer les données pour la réflexion
                system_age_hours = self.system_clock["total_uptime"] / 3600

                # Évaluer les progrès et changements depuis la dernière réflexion
                learning_assessment = self._assess_learning_capabilities()
                emotional_assessment = None
                if self.emotional_module:
                    emotional_assessment = self._assess_emotional_state()

                # Générer la réflexion
                reflection = {
                    "timestamp": time.time(),
                    "type": "periodic",
                    "system_age_hours": system_age_hours,
                    "content": f"Réflexion périodique après {system_age_hours:.2f} heures de fonctionnement.",
                    "assessments": {
                        "learning": learning_assessment,
                        "emotional": emotional_assessment,
                        "metacognition": {
                            "self_awareness": self.metacognition_metrics["self_awareness_level"],
                            "adaptation": self.metacognition_metrics["adaptation_effectiveness"]
                        }
                    },
                    "insights": self._generate_insights()
                }

                self.reflection_journal.append(reflection)
                logger.info(f"Réflexion périodique complétée après {system_age_hours:.2f} heures de fonctionnement")

            def _assess_emotional_state(self):
                """Évalue l'état émotionnel si le module est disponible"""
                if not self.emotional_module:
                    return None

                dominant_emotion = self.emotional_module.get_dominant_emotion()
                emotional_trend = self.emotional_module.get_emotional_trend()

                return {
                    "dominant_emotion": dominant_emotion,
                    "trend": emotional_trend,
                    "state_description": self.emotional_module.get_emotional_state_description()
                }

            def _update_metacognition_metrics(self):
                """Met à jour les métriques de métacognition"""
                # Mise à jour du niveau de conscience de soi
                if len(self.reflection_journal) > 1:
                    # Augmenter progressivement avec l'expérience
                    self.metacognition_metrics["self_awareness_level"] = min(
                        0.95,
                        self.metacognition_metrics["self_awareness_level"] + 0.001
                    )

                # Mise à jour de l'efficacité d'adaptation
                if hasattr(self.learning_module,
                           'learning_metrics') and 'prediction_accuracy' in self.learning_module.learning_metrics:
                    accuracies = self.learning_module.learning_metrics['prediction_accuracy']
                    if len(accuracies) > 20:
                        # Comparer les performances récentes aux performances précédentes
                        recent = sum(list(accuracies)[-10:]) / 10
                        previous = sum(list(accuracies)[-20:-10]) / 10

                        # Si amélioration, augmenter l'efficacité d'adaptation
                        if recent > previous:
                            improvement = (recent - previous) * 2  # Facteur d'échelle
                            self.metacognition_metrics["adaptation_effectiveness"] = min(
                                0.95,
                                self.metacognition_metrics["adaptation_effectiveness"] + improvement * 0.1
                            )
                        else:
                            # Diminution légère si pas d'amélioration
                            self.metacognition_metrics["adaptation_effectiveness"] = max(
                                0.1,
                                self.metacognition_metrics["adaptation_effectiveness"] - 0.01
                            )

                # Mise à jour de l'intégration des connaissances
                if hasattr(self.learning_module, 'concept_library'):
                    concept_count = len(self.learning_module.concept_library)
                    if concept_count > 0:
                        stable_ratio = sum(1 for _, concept in self.learning_module.concept_library.items()
                                           if concept.get('stability', 0) > 0.7) / concept_count

                        self.metacognition_metrics["knowledge_integration_rate"] = min(
                            0.95,
                            0.3 + stable_ratio * 0.6  # Base + ratio de concepts stables
                        )

            def _update_cognitive_state(self):
                """Met à jour l'état cognitif actuel"""
                # Simuler des fluctuations d'attention et de charge cognitive
                # Dans un système réel, ces valeurs seraient dérivées de métriques concrètes

                # Charge cognitive basée sur l'activité récente et la complexité des tâches
                if hasattr(self.self_model, 'internal_model') and 'recent_activity' in self.self_model.internal_model:
                    activity = self.self_model.internal_model['recent_activity']
                    if 'complexity' in activity:
                        self.cognitive_state["cognitive_load"] = min(0.95, activity['complexity'] * 0.8 + 0.1)

                # Profondeur de traitement - simulée ici, idéalement basée sur des métriques réelles
                processing_trend = random.uniform(-0.05, 0.05)  # Petites fluctuations aléatoires
                self.cognitive_state["processing_depth"] = max(0.1, min(0.95,
                                                                        self.cognitive_state[
                                                                            "processing_depth"] + processing_trend))

                # Conscience du contexte - pourrait être dérivée de la qualité de l'historique et de la mémoire
                if hasattr(self.self_model, 'communication_interface') and hasattr(
                        self.self_model.communication_interface, 'context_memory'):
                    context_len = len(self.self_model.communication_interface.context_memory)
                    context_quality = min(1.0, context_len / 10)  # Supposer qu'un contexte de 10 éléments est optimal
                    self.cognitive_state["context_awareness"] = 0.3 + context_quality * 0.6  # Base + qualité

            def _generate_insights(self):
                """Génère des insights basés sur l'état actuel du système"""
                insights = []

                # Insight sur les capacités d'apprentissage
                if hasattr(self.learning_module,
                           'learning_metrics') and 'prediction_accuracy' in self.learning_module.learning_metrics:
                    accuracies = list(self.learning_module.learning_metrics['prediction_accuracy'])
                    if len(accuracies) > 20:
                        recent_trend = sum(accuracies[-5:]) / 5 - sum(accuracies[-10:-5]) / 5

                        if recent_trend > 0.05:
                            insights.append("L'apprentissage montre une amélioration significative récente.")
                        elif recent_trend < -0.05:
                            insights.append("Une baisse de performance d'apprentissage récente nécessite attention.")

                # Insight sur l'état émotionnel si disponible
                if self.emotional_module:
                    dominant = self.emotional_module.get_dominant_emotion()
                    if dominant["emotion"] == "concern" and dominant["value"] > 0.7:
                        insights.append(
                            "Niveau élevé d'inquiétude détecté. Vérification des sources de stress recommandée.")
                    elif dominant["emotion"] == "satisfaction" and dominant["value"] > 0.8:
                        insights.append(
                            "Niveau optimal de satisfaction. Conditions favorables pour l'exploration et l'apprentissage.")

                # Insight sur la métacognition
                if self.metacognition_metrics["self_awareness_level"] > 0.8:
                    insights.append(
                        "Développement significatif de la conscience de soi. Capacités métacognitives renforcées.")

                # Insight sur l'environnement si disponible
                if hasattr(self.self_model, 'internal_model') and 'environment' in self.self_model.internal_model:
                    env = self.self_model.internal_model['environment']
                    if 'ambient_conditions' in env and 'noise_level' in env['ambient_conditions']:
                        if env['ambient_conditions']['noise_level'] > 0.8:
                            insights.append(
                                "Environnement très bruyant détecté. Impact potentiel sur la perception auditive.")

                # Si aucun insight spécifique n'a été généré
                if not insights:
                    insights.append("Fonctionnement global stable. Aucun insight particulier à signaler.")

                return insights

            def get_self_awareness_report(self):
                """Génère un rapport complet sur la conscience de soi du système"""
                current_time = time.time()
                uptime_hours = (current_time - self.system_clock["startup_time"]) / 3600

                report = f"== Rapport d'Auto-Conscience - Temps de fonctionnement: {uptime_hours:.2f} heures ==\n\n"

                # 1. Capacités cognitives
                report += "1. CAPACITÉS COGNITIVES\n"
                report += f"   Niveau de conscience de soi: {self.metacognition_metrics['self_awareness_level']:.2f}\n"
                report += f"   Capacité d'adaptation: {self.metacognition_metrics['adaptation_effectiveness']:.2f}\n"
                report += f"   Intégration des connaissances: {self.metacognition_metrics['knowledge_integration_rate']:.2f}\n"
                report += f"   Confiance de raisonnement: {self.metacognition_metrics['reasoning_confidence']:.2f}\n\n"

                # 2. État cognitif actuel
                report += "2. ÉTAT COGNITIF ACTUEL\n"
                if self.cognitive_state["attention_focus"]:
                    report += f"   Focus d'attention: {self.cognitive_state['attention_focus']}\n"
                else:
                    report += "   Focus d'attention: Non défini\n"
                report += f"   Profondeur de traitement: {self.cognitive_state['processing_depth']:.2f}\n"


                report += f"   Niveau de confiance global: {self.cognitive_state['confidence_level']:.2f}\n"
                report += f"   Charge cognitive: {self.cognitive_state['cognitive_load']:.2f}/10\n"
                report += f"   Mode de pensée: {self.cognitive_state['thinking_mode']}\n"

                # Inclure les concepts actifs dans la mémoire de travail
                report += "   Concepts actifs en mémoire de travail:\n"
                if self.cognitive_state['active_concepts']:
                    for concept in self.cognitive_state['active_concepts'][
                                   :5]:  # Limiter à 5 concepts pour la lisibilité
                        report += f"     - {concept}\n"
                    if len(self.cognitive_state['active_concepts']) > 5:
                        report += f"     - ...et {len(self.cognitive_state['active_concepts']) - 5} autres\n"
                else:
                    report += "     - Aucun concept actif actuellement\n"

                # Ajouter les objectifs actifs
                report += "   Objectifs actifs:\n"
                if self.cognitive_state['goals']:
                    for goal in self.cognitive_state['goals']:
                        report += f"     - {goal['description']} (priorité: {goal['priority']}/10)\n"
                else:
                    report += "     - Aucun objectif actif actuellement\n"

                # Ajouter l'état émotionnel si disponible
                if 'emotional_state' in self.cognitive_state:
                    report += f"   État émotionnel dominant: {self.cognitive_state['emotional_state']['dominant']}\n"
                    report += f"   Valence émotionnelle: {self.cognitive_state['emotional_state']['valence']:.2f}\n"
                    report += f"   Intensité émotionnelle: {self.cognitive_state['emotional_state']['intensity']:.2f}\n"

                # Ajouter l'état d'introspection
                report += f"   Niveau d'introspection: {self.cognitive_state['introspection_level']:.2f}\n"
                report += f"   Temps depuis dernière phase d'introspection: {self.cognitive_state['time_since_introspection']} sec\n"

if __name__ == "__main__":
    system = SelfAwareSystem()  # ou EnhancedSelfAwareSystem()
    if system.start():
        print("Système démarré, appuie sur Ctrl+C pour arrêter.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            system.stop()
    else:
        print("Échec du démarrage du système.")