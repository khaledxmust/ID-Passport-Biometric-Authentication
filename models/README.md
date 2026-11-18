# External document models

The original service referenced two local Ultralytics segmentation weights, `SegID.pt` and `SegPS.pt`. They are excluded from this prepared source copy because their redistribution terms and training data provenance are unconfirmed.

Obtain authorized copies and configure `ID_MODEL_PATH` and `PASSPORT_MODEL_PATH`. The expected model task is segmentation. The API rectifies the first detected polygon. A generic object detector with bounding boxes alone cannot replace these weights.

Models trained with Ultralytics 8.0.x may need a matching conversion environment before they load in the proposed 8.3.x environment. Loading the historical checkpoints has not been tested here. Load only trusted checkpoint files.

Document OCR with `detection=false`, manual TD3 parsing, and the health route do not use these document models. DeepFace has its own external face/detector weights and can download them on its first face report. Review the library and model licenses before redistribution.
