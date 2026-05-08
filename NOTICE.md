# Notices

FaceGate is a standalone folder comparison tool built with InsightFace.

## Inspired By

Face Analysis for ComfyUI:

https://github.com/cubiq/ComfyUI_FaceAnalysis

The standalone implementation in this repository was informed by that custom
node's use of InsightFace embeddings and cosine-distance face comparison.

## Dependencies

InsightFace:

https://github.com/deepinsight/insightface

ONNX Runtime:

https://onnxruntime.ai/

## Model Notice

This project does not include the `antelopev2` ONNX model files. Users must
download and place them under:

```text
models/insightface/models/antelopev2/
```

InsightFace documentation notes that its provided model packs are for
non-commercial research purposes. Check the current upstream terms before
redistribution or commercial use.
