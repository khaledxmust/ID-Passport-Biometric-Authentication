"""Request validation without importing image or ML dependencies."""

import json
from urllib.parse import urlparse

SUPPORTED_REPORTS = (
    'face_similarity', 'face_representation', 'mrz_manual_extraction',
    'passporteye_verification', 'passocr_verification', 'id_eg_extraction', 'facial_analysis',
)
DOCUMENT_REPORTS = {'face_similarity', 'passporteye_verification', 'passocr_verification', 'id_eg_extraction'}


def parse_request(body, allow_webhooks=False, allow_facial_analysis=False):
    if not isinstance(body, dict) or 'request' not in body:
        raise ValueError('Pass a JSON object containing a request object.')
    value = body['request']
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError('The request string must be valid JSON.') from error
    if not isinstance(value, dict):
        raise ValueError('request must be a JSON object.')
    reference = value.get('merchantReference')
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError('merchantReference must be a nonempty string.')
    reports = value.get('reports')
    if not isinstance(reports, list) or not reports or not all(isinstance(report, str) for report in reports):
        raise ValueError('reports must be a nonempty list of report names.')
    unknown = set(reports) - set(SUPPORTED_REPORTS)
    if unknown:
        raise ValueError('Unsupported reports: ' + ', '.join(sorted(unknown)))
    if 'facial_analysis' in reports and not allow_facial_analysis:
        raise ValueError('The legacy facial_analysis report is disabled. Configure ALLOW_FACIAL_ANALYSIS to enable it.')
    if DOCUMENT_REPORTS.intersection(reports) and not value.get('documentImage'):
        raise ValueError('documentImage is required for the selected reports.')
    if 'face_similarity' in reports and not (value.get('imageLink') or value.get('videoLink')):
        raise ValueError('face_similarity needs imageLink or videoLink.')
    if value.get('imageLink') and value.get('videoLink'):
        raise ValueError('Provide either imageLink or videoLink per request.')
    if {'face_representation', 'facial_analysis'}.intersection(reports) and not value.get('imageLink'):
        raise ValueError('The selected face report needs imageLink.')
    if 'mrz_manual_extraction' in reports and not isinstance(value.get('mrz'), str):
        raise ValueError('mrz_manual_extraction needs an MRZ string.')
    if 'id_eg_extraction' in reports and value.get('document_face') not in ('front', 'back'):
        raise ValueError('document_face must be front or back.')
    detection = value.get('detection', False)
    if isinstance(detection, str) and detection.lower() in ('true', 'false'):
        detection = detection.lower() == 'true'
    if not isinstance(detection, bool):
        raise ValueError('detection must be true or false.')
    value = dict(value, detection=detection)
    for name in ('documentImage', 'imageLink', 'videoLink', 'hookLink'):
        url = value.get(name)
        if url:
            parsed = urlparse(url) if isinstance(url, str) else None
            if (parsed is None or parsed.scheme not in ('http', 'https') or not parsed.hostname
                    or parsed.username is not None or parsed.password is not None):
                raise ValueError(name + ' must be an HTTP or HTTPS URL without credentials.')
    if value.get('hookLink') and not allow_webhooks:
        raise ValueError('Webhooks are disabled. Configure ALLOW_WEBHOOKS to use hookLink.')
    return value
