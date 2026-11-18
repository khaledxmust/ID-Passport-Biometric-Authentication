"""TD3 passport MRZ parsing and two OCR paths."""

import io
import config
from src.v1 import Model


def _parse_mrz(text):
    from mrz.checker.td3 import TD3CodeChecker

    normalized = '\n'.join(line.strip() for line in text.strip().splitlines() if line.strip())
    lines = normalized.splitlines()
    if len(lines) != 2 or any(len(line) != 44 for line in lines):
        raise ValueError('A TD3 MRZ requires two lines of 44 characters.')
    check = TD3CodeChecker(normalized)
    fields = check.fields()
    return dict(fields._asdict(), verification=bool(check), raw_text=normalized)


def mrz_manual_extraction(data):
    return {'reportName': 'mrz_manual_extraction', 'result': {
        'stat': 'done', 'message': _parse_mrz(data['mrz'])}}


def passporteye_verification(data):
    import cv2
    from passporteye import read_mrz

    ok, encoded = cv2.imencode('.jpg', data['doc'])
    if not ok:
        raise ValueError('Could not encode the document image.')
    mrz = read_mrz(io.BytesIO(encoded.tobytes()))
    if mrz is None:
        raise ValueError('No passport MRZ was found.')
    return {'reportName': 'passporteye_verification', 'result': {'stat': 'done', 'message': dict(mrz.to_dict())}}


def passocr_verification(data):
    import cv2
    import pytesseract

    image = Model.segment_document(data['doc'], 'passport') if data['detection'] else data['doc']
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    height, width = image.shape[:2]
    cropped = image[int(height * 0.76):int(height * 0.94), :width]
    text = pytesseract.image_to_string(cropped, lang=config.MRZ_LANGUAGE, config='--oem 3 --psm 6')
    # Tesseract frequently inserts spaces inside the two MRZ lines.
    text = '\n'.join(''.join(line.split()) for line in text.splitlines() if line.strip())
    return {'reportName': 'passocr_verification', 'result': {'stat': 'done', 'message': _parse_mrz(text)}}
