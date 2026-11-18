"""Portable settings for the local biometric API."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def model_path(variable, filename):
    value = Path(os.getenv(variable, str(ROOT / 'models' / filename))).expanduser()
    return value if value.is_absolute() else ROOT / value


class Configuration:
    MAX_CONTENT_LENGTH = 1024 * 1024
    JSON_SORT_KEYS = False
    MEDIA_TIMEOUT = float(os.getenv('MEDIA_TIMEOUT', '15'))
    MAX_MEDIA_BYTES = int(os.getenv('MAX_MEDIA_BYTES', str(20 * 1024 * 1024)))
    ALLOW_WEBHOOKS = os.getenv('ALLOW_WEBHOOKS', 'false').lower() == 'true'
    ALLOW_FACIAL_ANALYSIS = os.getenv('ALLOW_FACIAL_ANALYSIS', 'false').lower() == 'true'


ID_MODEL_PATH = model_path('ID_MODEL_PATH', 'id-segmentation.pt')
PASSPORT_MODEL_PATH = model_path('PASSPORT_MODEL_PATH', 'passport-segmentation.pt')
FACE_MODEL = os.getenv('FACE_MODEL', 'VGG-Face')
FACE_DETECTOR = os.getenv('FACE_DETECTOR', 'retinaface')
FACE_DISTANCE = os.getenv('FACE_DISTANCE', 'euclidean_l2')
ID_TEXT_LANGUAGE = os.getenv('ID_TEXT_LANGUAGE', 'ara')
ID_NUMBER_LANGUAGE = os.getenv('ID_NUMBER_LANGUAGE', 'eng')
MRZ_LANGUAGE = os.getenv('MRZ_LANGUAGE', 'eng')
