"""Fixed-layout Egyptian ID OCR retained from the original implementation."""

import config
from src.v1 import Model
from src.v1.Functions import frontID, backID


def id_eg_extraction(data):
    image = Model.segment_document(data['doc'], 'id') if data['detection'] else data['doc']
    languages = [config.ID_NUMBER_LANGUAGE, config.ID_TEXT_LANGUAGE, config.ID_TEXT_LANGUAGE]
    extract = frontID if data['document_face'] == 'front' else backID
    return {'reportName': 'id_eg_extraction', 'result': {'stat': 'done', 'message': extract(image, languages)}}
