"""Dispatch validated report requests with structured JSON responses."""

import json
from flask import Blueprint, current_app, request
from src.v1 import Model
from src.v1.validation import parse_request

biometric_v1 = Blueprint('biometric_v1', __name__)


def _handler(name):
    # Import each adapter only when needed. The health and MRZ paths do not load ML models.
    if name in ('face_similarity', 'face_representation', 'facial_analysis'):
        from src.v1.routex import ver
        return getattr(ver, name)
    if name == 'id_eg_extraction':
        from src.v1.routex.id import id_eg_extraction
        return id_eg_extraction
    from src.v1.routex import ppas
    return getattr(ppas, name)


def _json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError('Report contains an unsupported JSON value.')


@biometric_v1.post('/reports')
def reports():
    try:
        data = parse_request(request.get_json(silent=True), current_app.config['ALLOW_WEBHOOKS'],
            current_app.config['ALLOW_FACIAL_ANALYSIS'])
    except ValueError as error:
        return {'error': str(error)}, 400
    limits = {'timeout': current_app.config['MEDIA_TIMEOUT'], 'max_bytes': current_app.config['MAX_MEDIA_BYTES']}
    prepared = dict(data, doc=None, img=None, vid=False)
    try:
        if data.get('documentImage'):
            prepared['doc'] = Model.requestImage(data['documentImage'], **limits)
        if data.get('imageLink'):
            prepared['img'] = Model.requestImage(data['imageLink'], **limits)
        elif data.get('videoLink'):
            prepared['img'] = Model.extractImages(data['videoLink'], **limits)
            prepared['vid'] = True
    except Exception:
        return {'error': 'Could not read the supplied media. Check the direct URL, size, and media format.'}, 422

    result = {'merchantReference': data['merchantReference'], 'reportResult': []}
    for name in data['reports']:
        try:
            report = _handler(name)(prepared)
            report = json.loads(json.dumps(report, default=_json_default, allow_nan=False))
        except (ImportError, RuntimeError) as error:
            report = {'reportName': name, 'result': {'stat': 'failed', 'message': str(error)}}
        except Exception:
            report = {'reportName': name, 'result': {'stat': 'failed', 'message': 'Report could not process the supplied input.'}}
        result['reportResult'].append(report)
    if data.get('hookLink'):
        import requests
        try:
            response = requests.post(data['hookLink'], json=result, timeout=limits['timeout'], allow_redirects=False)
            result['webhookDelivered'] = 200 <= response.status_code < 300
        except requests.RequestException:
            result['webhookDelivered'] = False
    return result
