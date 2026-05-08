# FaceGate

Batch sort images by face similarity using InsightFace `antelopev2` and cosine
distance.

The tool compares every image in a target folder against the reference images
in `reference_image/`. It uses a best-match decision: a candidate passes if its
best cosine distance to any usable reference image is less than or equal to the
threshold.

## What It Does

- Uses InsightFace `FaceAnalysis(name="antelopev2")`.
- Prefers `CUDAExecutionProvider` and falls back to CPU.
- Supports multiple reference images.
- Copies accepted images to `passed_comparison/`.
- Copies rejected images to `failed_comparison/`.
- Writes `comparison_report.csv`.
- Copies files with `shutil.copy2`, so output images are not resized,
  recompressed, or re-encoded.

Back-of-head images are not useful references for this tool. InsightFace needs a
detectable face. Front, three-quarter, and profile face references are useful.

## Folder Layout

```text
face-folder-comparison/
  compare_faces.py
  run_face_compare.bat
  setup.bat
  requirements.txt
  reference_image/
  insightface/
    models/
      antelopev2/
  passed_comparison/
  failed_comparison/
```

## Setup

Install Python 3.10 or 3.11, then run:

```bat
setup.bat
```

The setup script creates a local `.venv` inside this project and installs the
Python dependencies there. It does not use or modify any ComfyUI environment.

## Model Files

Put the five `antelopev2` ONNX files here:

```text
insightface\models\antelopev2
```

The `models\antelopev2` part is required by InsightFace. This project uses
`insightface` as the model root, so the full path stays compact.

Expected files:

```text
1k3d68.onnx
2d106det.onnx
genderage.onnx
glintr100.onnx
scrfd_10g_bnkps.onnx
```

References:

- InsightFace PyPI model zoo: https://pypi.org/project/insightface/
- Hugging Face mirror: https://huggingface.co/fofr/comfyui/tree/main/insightface/models/antelopev2
- Alternative Hugging Face mirror: https://huggingface.co/lithiumice/insightface/tree/main/models/antelopev2

InsightFace documentation notes that these model packs are for
non-commercial research purposes. Check the current model terms before
redistributing models or using them commercially.

## Run

1. Put one or more reference face images in `reference_image/`.
2. Double-click `run_face_compare.bat`.
3. Enter the folder you want to compare.
4. Enter a cosine distance threshold.

Lower thresholds are stricter. A common starting point is:

```text
0.68
```

## Command Line

```bat
.venv\Scripts\python.exe compare_faces.py ^
  --input-folder "G:\path\to\images" ^
  --threshold 0.68 ^
  --clear-output
```

Useful options:

```text
--input-folder PATH   Folder of images to compare
--threshold FLOAT     Cosine distance threshold
--clear-output        Empty passed/failed folders before running
--no-recursive        Do not scan subfolders
--model-root PATH     Override the InsightFace model root
```

`--model-root` should point to the folder that contains `models\antelopev2`,
not directly to the `antelopev2` folder.

## Optional Environment Variables

```text
FACE_COMPARE_MODEL_ROOT
```

Overrides the model root. Expected layout:

```text
%FACE_COMPARE_MODEL_ROOT%\models\antelopev2\*.onnx
```

```text
FACE_COMPARE_CUDA_DLL_DIRS
```

Windows-only. Optional semicolon-separated list of CUDA/cuDNN DLL folders to
add to the process before ONNX Runtime loads. This is useful if
`onnxruntime-gpu` is installed but CUDA DLLs are not on the system path.

Example:

```bat
set FACE_COMPARE_CUDA_DLL_DIRS=C:\path\to\torch\lib
```

## Report

`comparison_report.csv` contains:

```text
source
status
cosine_distance
cosine_similarity
best_reference
face_count
message
output_path
```

Images with no detected face are copied to `failed_comparison/`.

## Credits

This project was inspired by the Face Analysis for ComfyUI custom node:

https://github.com/cubiq/ComfyUI_FaceAnalysis

The implementation here is standalone and focused on folder-based image
sorting, but it follows the same general InsightFace embedding and
cosine-distance comparison approach.

InsightFace is developed by DeepInsight:

https://github.com/deepinsight/insightface
