"""Face comparison and representation adapters."""

import config
from src.v1 import Model


def face_similarity(data):
    if data['vid']:
        result = Model.videoVerification(data['doc'], data['img'])
    else:
        result = Model.imageVerification(data['doc'], data['img'])
    return {'reportName': 'face_similarity', 'result': {'stat': 'done',
        'verified': bool(result['verified']), 'message': result}}


def face_representation(data):
    return {'reportName': 'face_representation', 'result': {
        'stat': 'done', 'message': Model.faceEmbedding(data['img'])}}


def facial_analysis(data):
    """Retain the original attribute-analysis report behind explicit configuration."""
    result = Model.analyze(data['img'], ['age', 'gender', 'emotion', 'race'], config.FACE_DETECTOR)
    return {'reportName': 'facial_analysis', 'result': {'stat': 'done', 'message': result}}
