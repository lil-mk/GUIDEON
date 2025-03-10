# sensors.py
# Gère les capteurs (caméra, micro, GPS)

import cv2
import speech_recognition as sr
import logging
import requests
import time

logger = logging.getLogger("Sensors")

class SensorModule:
    def __init__(self, use_camera=True, use_microphone=True):
        self.use_camera = use_camera
        self.use_microphone = use_microphone
        self.camera = None
        self.recognizer = None
        self.microphone = None

        self._initialize_sensors()

    def _initialize_sensors(self):
        """Initialise les capteurs"""
        if self.use_camera:
            self.camera = cv2.VideoCapture(0)

        if self.use_microphone:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()

        try:
            requests.get("https://www.google.com", timeout=3)
        except:
            pass

    def capture_frame(self):
        if self.camera:
            ret, frame = self.camera.read()
            return frame if ret else None
        return None

    def listen(self, timeout=5):
        if not self.microphone:
            return ""
        with self.microphone as source:
            audio = self.recognizer.listen(source, timeout=timeout)
            try:
                return self.recognizer.recognize_google(audio, language="fr-FR")
            except:
                return ""

    def release_resources(self):
        if self.camera:
            self.camera.release()