# perception.py
# Analyse des données des capteurs

import cv2
import face_recognition
import numpy as np
import logging

logger = logging.getLogger("Perception")

class PerceptionModule:
    def detect_faces(self, frame):
        """Détecte les visages"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return face_recognition.face_locations(rgb_frame)