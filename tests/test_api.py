"""Small regression checks. Model inference is mocked and no media is downloaded."""

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from main import create_app
from src.v1 import Model
from src.v1.validation import parse_request

ROOT = Path(__file__).resolve().parents[1]


class RequestValidationTests(unittest.TestCase):
    def setUp(self):
        self.body = json.loads((ROOT / 'examples/manual_mrz.json').read_text())

    def test_object_and_json_string_inputs(self):
        expected = parse_request(self.body)
        self.assertEqual(expected, parse_request({'request': json.dumps(self.body['request'])}))

    def test_invalid_request_shapes(self):
        for body in [None, {}, {'request': []}, {'request': 'not JSON'}]:
            with self.subTest(body=body), self.assertRaises(ValueError):
                parse_request(body)

    def test_required_reference_and_reports(self):
        for name, value in [('merchantReference', None), ('reports', []), ('reports', ['unknown'])]:
            body = copy.deepcopy(self.body)
            body['request'][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                parse_request(body)

    def test_document_and_face_requirements(self):
        for report in ['face_similarity', 'face_representation', 'passporteye_verification', 'id_eg_extraction']:
            body = copy.deepcopy(self.body)
            body['request']['reports'] = [report]
            with self.subTest(report=report), self.assertRaises(ValueError):
                parse_request(body)

    def test_detection_false_is_respected(self):
        body = json.loads((ROOT / 'examples/id_ocr.json').read_text())
        body['request']['detection'] = 'false'
        self.assertIs(parse_request(body)['detection'], False)

    def test_legacy_facial_analysis_is_opt_in(self):
        body = json.loads((ROOT / 'examples/face_comparison.json').read_text())
        body['request']['reports'] = ['facial_analysis']
        with self.assertRaises(ValueError):
            parse_request(body)
        self.assertEqual(parse_request(body, allow_facial_analysis=True)['reports'], ['facial_analysis'])

    def test_urls_and_webhooks(self):
        for url in ['file:///etc/passwd', 'https://user:password@example.com/x', 4]:
            body = copy.deepcopy(self.body)
            body['request']['documentImage'] = url
            with self.subTest(url=url), self.assertRaises(ValueError):
                parse_request(body)
        body = copy.deepcopy(self.body)
        body['request']['hookLink'] = 'https://example.com/callback'
        with self.assertRaises(ValueError):
            parse_request(body)


class RouteTests(unittest.TestCase):
    def setUp(self):
        app = create_app()
        app.config.update(TESTING=True, ALLOW_WEBHOOKS=False)
        self.client = app.test_client()

    def test_health_without_models(self):
        self.assertEqual(self.client.get('/health').status_code, 200)

    def test_bad_json_is_a_client_error(self):
        response = self.client.post('/biometric/v1/reports', data='broken', content_type='application/json')
        self.assertEqual(response.status_code, 400)

    def test_structured_report_and_boolean(self):
        body = json.loads((ROOT / 'examples/face_comparison.json').read_text())
        with patch.object(Model, 'requestImage', return_value='image'), patch.object(
            Model, 'imageVerification', return_value={'verified': False, 'distance': 0.8}
        ):
            response = self.client.post('/biometric/v1/reports', json=body)
        self.assertEqual(response.status_code, 200)
        report = response.json['reportResult'][0]
        self.assertIs(report['result']['verified'], False)
        self.assertIsInstance(response.json['reportResult'], list)

    def test_media_failure_has_controlled_response(self):
        body = json.loads((ROOT / 'examples/face_comparison.json').read_text())
        with patch.object(Model, 'requestImage', side_effect=ValueError('bad media')):
            response = self.client.post('/biometric/v1/reports', json=body)
        self.assertEqual(response.status_code, 422)

    def test_missing_dependency_is_report_failure(self):
        body = json.loads((ROOT / 'examples/face_comparison.json').read_text())
        with patch.object(Model, 'requestImage', return_value='image'), patch.object(
            Model, 'imageVerification', side_effect=RuntimeError('Missing test dependency')
        ):
            response = self.client.post('/biometric/v1/reports', json=body)
        self.assertEqual(response.json['reportResult'][0]['result']['stat'], 'failed')

    def test_actual_manual_mrz_without_media_or_model(self):
        try:
            import mrz  # noqa: F401
        except ImportError:
            self.skipTest('Install base requirements to exercise the actual MRZ parser.')
        body = json.loads((ROOT / 'examples/manual_mrz.json').read_text())
        with patch.object(Model, 'requestImage', side_effect=AssertionError('No media needed')):
            response = self.client.post('/biometric/v1/reports', json=body)
        report = response.json['reportResult'][0]['result']
        self.assertEqual(report['stat'], 'done')
        self.assertTrue(report['message']['verification'])


class VideoRegressionTests(unittest.TestCase):
    def test_best_frame_keeps_its_verified_flag(self):
        first = {'distance': 0.2, 'verified': True}
        last = {'distance': 0.9, 'verified': False}
        with patch.object(Model, 'imageVerification', side_effect=[first, last]):
            result = Model.videoVerification('document', ['first-frame', 'last-frame'])
        self.assertEqual(result['distance'], 0.2)
        self.assertIs(result['verified'], True)
        self.assertEqual(result['frameIndex'], 0)

    def test_unusable_video_frames_fail(self):
        with patch.object(Model, 'imageVerification', side_effect=ValueError('No face')):
            with self.assertRaises(ValueError):
                Model.videoVerification('document', ['frame'])


if __name__ == '__main__':
    unittest.main()
