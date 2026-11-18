# ID/Passport Biometric Authentication

A Flask API that combines face comparison, passport MRZ parsing, Egyptian ID OCR, and optional document segmentation. It assembles pretrained face and document models with perspective correction and fixed document-layout crops.

The project exposes the existing report-oriented API through `POST /biometric/v1/reports`. Its six default identity/document report handlers and optional legacy handler are listed below.

## Processing flow

```text
JSON request
    -> validation and bounded media download
    -> optional YOLO document segmentation
    -> perspective correction and fixed-layout OCR / face comparison / TD3 checker
    -> structured JSON report results
```

| Report | Inputs | External requirements |
| --- | --- | --- |
| `face_similarity` | `documentImage` and `imageLink` or `videoLink` | DeepFace, face/detector weights |
| `face_representation` | `imageLink` | DeepFace, face/detector weights |
| `mrz_manual_extraction` | `mrz`, two 44-character TD3 lines | `mrz` Python package |
| `passporteye_verification` | `documentImage` | PassportEye and Tesseract |
| `passocr_verification` | `documentImage`, optional `detection` | OpenCV, Tesseract, optional passport segmentation weights |
| `id_eg_extraction` | `documentImage`, `document_face`, optional `detection` | OpenCV, Tesseract, optional ID segmentation weights |
| `facial_analysis` | `imageLink` and explicit `ALLOW_FACIAL_ANALYSIS=true` configuration | DeepFace and attribute-analysis weights. Disabled by default. |

## Setup

Use Python 3.10 or 3.11 for the proposed model dependency baseline. The web layer can be exercised separately from the model stack. The original project used Ultralytics 8.0.20 and DeepFace 0.0.79. This copy uses a proposed dependency range compatible with the current adapter interfaces, and full checkpoint compatibility still needs verification.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python main.py
```

The server listens on `http://127.0.0.1:8016` by default. `GET /health` confirms that the API starts. It does not load or validate ML models.

For image OCR, install the optional Python dependencies and install the Tesseract executable separately.

```powershell
python -m pip install -r requirements-ocr.txt
tesseract --list-langs
```

The default OCR languages are `ara` for Arabic text and `eng` for numbers and passport MRZs. The historical implementation used custom `ara_number`, `Arabic`, and `ckbLayer_fast` traineddata. Those custom assets are excluded from this copy. Configurable standard languages let the pipeline run with ordinary Tesseract installation, but extraction quality may differ. Configure the original language names only when authorized traineddata files are installed. On Windows set `TESSERACT_CMD` if the executable is absent from PATH. The optional `bdate` OCR result preserves six encoded ID characters and has no calendar/date validation.

For face comparison or YOLO document segmentation, install the model dependencies in the Python 3.10/3.11 environment.

```powershell
python -m pip install -r requirements-models.txt
```

Set `ID_MODEL_PATH` and `PASSPORT_MODEL_PATH` to authorized local segmentation checkpoints. See `models/README.md`. Face models use `VGG-Face`, `retinaface`, and `euclidean_l2` by default, matching the effective settings in the original model wrapper. DeepFace may download its model weights on first use. This is separate from the document checkpoints, whose automatic Google Drive downloads have been removed.

## Try a report

The synthetic TD3 example can be exercised with the base dependencies. Its fictional UTO example data is used for parser testing.

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8016/biometric/v1/reports `
  -ContentType application/json `
  -Body (Get-Content examples/manual_mrz.json -Raw)
```

`examples/face_comparison.json` and `examples/id_ocr.json` contain placeholder URLs. Replace them with direct URLs for consented or synthetic demonstration media. Media URLs must use HTTP or HTTPS and must return the file directly. Redirects are disabled. For video, the service downloads a bounded temporary copy, samples up to four frames approximately one second apart, and removes the temporary file. The full comparison result comes from the sampled frame with the smallest distance.

Example request shape

```json
{
  "request": {
    "merchantReference": "demo-001",
    "reports": ["id_eg_extraction"],
    "documentImage": "https://example.com/synthetic-id-front.jpg",
    "document_face": "front",
    "detection": false
  }
}
```

The response contains `merchantReference` and a JSON array named `reportResult`. Each report carries `reportName` and `result`. A completed comparison returns `verified` as a JSON boolean. A model or OCR failure is reported within that report, so callers should inspect every `result.stat`. Invalid requests use HTTP 400. Unreadable media uses HTTP 422. HTTP 200 can contain failed individual reports.

The API also accepts a JSON-encoded string in `request` for older callers. Python dictionary strings with single quotes are unsupported. The original stringified `reportResult` is now a JSON array. Clients relying on the string representation need updating.

`hookLink` is available only when `ALLOW_WEBHOOKS=true`. A callback failure is returned as `webhookDelivered=false` and does not discard the completed reports. Run with trusted inputs in a local environment. Public exposure requires authentication, URL access controls, request throttling, and a privacy/retention design.

## Evaluation and limitations

- Face comparison integrates pretrained models. The repository does not establish a measured false-match rate, false-nonmatch rate, demographic fairness, or an identity assurance level.
- Sampling video frames performs face similarity. It does not implement liveness or presentation-attack detection.
- MRZ check digits test formatting and consistency. They do not establish that a document was issued by an authority or that the holder is authorized to use it.
- Egyptian ID OCR uses fixed normalized crop coordinates and depends on document alignment, layout, image resolution, and installed traineddata. It has no calibrated field-accuracy benchmark in this source copy.
- Segmentation uses the first returned document polygon. Multiple documents, upside-down content, severe perspective distortion, and unknown layouts need further handling.
- The original `facial_analysis` adapter is retained for compatibility behind `ALLOW_FACIAL_ANALYSIS=false`. It requests the historical age, gender, emotion, and race attributes. The default identity/document demo leaves this disabled. No attribute inference was run or evaluated while preparing this copy.

## Checks

```powershell
python -m unittest discover -s tests -v
```

The regression checks use the real Flask API with mocked media/model adapters. They cover validation, structured results, an actual synthetic MRZ parse when `mrz` is installed, media errors, and the best-video-frame result. Full model inference, Tesseract extraction, and historical checkpoint loading still need a separate environment and authorized test assets. Dependency ranges are a proposed setup, not a frozen environment validated for every optional library.

## Source and assets

This folder was prepared from the local `PycharmProjects/biometric` implementation and its matching `xPycharmProjects` copy. The report API, ID crop logic, MRZ handling, and model integration originate there. Cleanup removed circular imports, import-time downloads, machine/cloud configuration, unused Swagger scaffolding, credential files, historical report data, and bundled checkpoints.

The prepared copy fixes request validation, uninitialized media variables, structured JSON responses, boolean parsing, BGR consistency, complete best-frame video results, capture cleanup, segmentation API compatibility, ID crop-axis errors, and color-preserving document correction.

No code license is asserted in this folder. Confirm the right to publish the implementation and select an appropriate license before upload. External libraries and model weights retain their own terms.

Relevant upstream references

- [DeepFace model and verification documentation](https://github.com/serengil/deepface)
- [Ultralytics segmentation result API](https://docs.ultralytics.com/modes/predict/)
- [PassportEye](https://pypi.org/project/PassportEye/)
- [MRZ checker](https://pypi.org/project/mrz/)
- [Tesseract](https://github.com/tesseract-ocr/tesseract)
