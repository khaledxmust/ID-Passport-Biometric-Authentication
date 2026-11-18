"""Media loading and lazy adapters around external models.

Images remain in OpenCV BGR format throughout the pipeline.
"""

import tempfile
from functools import lru_cache
from pathlib import Path

import config


def _download(url, timeout=15, max_bytes=20 * 1024 * 1024):
    import requests

    with requests.get(url, stream=True, timeout=timeout, allow_redirects=False) as response:
        response.raise_for_status()
        if 300 <= response.status_code < 400:
            raise ValueError('Use a direct media URL. Redirects are disabled.')
        content = bytearray()
        for chunk in response.iter_content(65536):
            content.extend(chunk)
            if len(content) > max_bytes:
                raise ValueError('Media exceeds MAX_MEDIA_BYTES.')
    return bytes(content)


def requestImage(url, **limits):
    import cv2
    import numpy as np

    data = _download(url, **limits)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('The URL did not return a readable image.')
    return image


def extractImages(url, **limits):
    """Sample at most four frames, approximately one second apart."""
    import cv2

    data = _download(url, **limits)
    path = None
    capture = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as handle:
            path = Path(handle.name)
            handle.write(data)
        capture = cv2.VideoCapture(str(path))
        frames = []
        for second in range(4):
            capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)
        if not frames:
            raise ValueError('The URL did not return readable video frames.')
        return frames
    finally:
        if capture is not None:
            capture.release()
        if path is not None:
            path.unlink(missing_ok=True)


def _deepface():
    try:
        from deepface import DeepFace
    except ImportError as error:
        raise RuntimeError('Face reports require requirements-models.txt and the configured face weights.') from error
    return DeepFace


def represent(img_path, model_name, detector_backend, enforce_detection=True, align=True):
    return _deepface().represent(img_path=img_path, model_name=model_name,
        detector_backend=detector_backend, enforce_detection=enforce_detection, align=align)


def verify(img1_path, img2_path, model_name, detector_backend, distance_metric,
           enforce_detection=True, align=True):
    return _deepface().verify(img1_path=img1_path, img2_path=img2_path, model_name=model_name,
        detector_backend=detector_backend, distance_metric=distance_metric,
        enforce_detection=enforce_detection, align=align)


def faceEmbedding(img):
    return represent(img, config.FACE_MODEL, config.FACE_DETECTOR)


def analyze(img_path, actions, detector_backend, enforce_detection=True, align=True):
    """Legacy adapter, reachable only when facial analysis is explicitly enabled."""
    return _deepface().analyze(img_path=img_path, actions=actions,
        detector_backend=detector_backend, enforce_detection=enforce_detection, align=align)


def imageVerification(img1, img2):
    return verify(img1, img2, config.FACE_MODEL, config.FACE_DETECTOR, config.FACE_DISTANCE)


def videoVerification(document, frames):
    """Return the complete result of the lowest-distance usable frame."""
    results = []
    for index, frame in enumerate(frames):
        try:
            result = imageVerification(document, frame)
        except ValueError:
            continue  # A frame with no detected face can be skipped.
        result = dict(result, frameIndex=index)
        results.append(result)
    if not results:
        raise ValueError('No usable face could be compared in the sampled frames.')
    return min(results, key=lambda result: float(result['distance']))


@lru_cache(maxsize=2)
def _segmentation_model(kind):
    path = config.ID_MODEL_PATH if kind == 'id' else config.PASSPORT_MODEL_PATH
    if not path.is_file():
        raise RuntimeError(f'Missing {kind} segmentation weights. Configure the model path using .env.example.')
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError('Document detection requires requirements-models.txt.') from error
    return YOLO(str(path))


def segment_document(image, kind):
    from src.v1.Functions import imgMasking

    result = _segmentation_model(kind).predict(image, conf=0.25, retina_masks=True, verbose=False)[0]
    masks = result.masks
    if masks is None or len(masks) == 0:
        raise ValueError('No document segmentation was found.')
    if hasattr(masks, 'xy'):
        polygon = masks.xy[0]
        normalized = False
    else:
        polygon = masks.segments[0]
        normalized = True
    return imgMasking(image, polygon, normalized=normalized)
