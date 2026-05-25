# FaceGate ComfyUI Nodes

This folder contains the ComfyUI custom node implementation for FaceGate.

## Install

Copy or symlink this `comfyui_nodes` folder into ComfyUI's `custom_nodes`
folder. You can rename the copied folder to `FaceGate` if you prefer:

```text
ComfyUI/
  custom_nodes/
    FaceGate/
      __init__.py
      facegate_comfyui.py
```

Install the Python dependencies into the same Python environment that runs
ComfyUI:

```bat
python -m pip install insightface==0.7.3 onnxruntime-gpu numpy pillow
```

If you use ComfyUI's portable Windows build, run that command with its embedded
Python executable.

## Model Files

FaceGate uses InsightFace `antelopev2`. The node searches these model roots:

```text
<model_root input>
ComfyUI folder_paths.get_folder_paths("insightface")
<ComfyUI>/models/insightface
<ComfyUI>/models/insightface/models
<ComfyUI>/models/insightface/models/antelopev2
<ComfyUI>/models/insightface/antelopev2
<ComfyUI>/models
```

The expected layout is:

```text
<model-root>/models/antelopev2/*.onnx
```

The direct ComfyUI layout below is also supported:

```text
<ComfyUI>/models/insightface/antelopev2/*.onnx
```

## Node

Display name: `FaceGate Filter`

Inputs:

```text
images: IMAGE
reference_folder: STRING
threshold: FLOAT
```

`reference_folder` must be set to your own folder of reference face images. The
node does not use the standalone FaceGate `reference_image` folder.

Outputs:

```text
passed_images: IMAGE
failed_images: IMAGE
pass_mask: MASK
scores: STRING
debug_images: IMAGE
pass_mask_image: IMAGE
```

`pass_mask` is white for input images that passed and black for input images
that failed. `pass_mask_image` is the same pass/fail information as a visible
black/white image batch, useful when a ComfyUI mask viewer makes mask values
ambiguous. `scores` is JSON containing distance, similarity, best reference,
face count, pass/fail reason, and warning details.

## Save Passed Images

Display name: `Facegate Save Image`

Use this instead of ComfyUI's built-in Save Image when you want failed images
discarded. If `scores` is not connected, this node behaves like normal Save
Image and saves every image it receives. To let it skip failed images, connect
FaceGate's `scores` output to the optional link-only `scores` socket. If no
images passed, this node saves nothing and does not create a black placeholder. Its
`filename_prefix` input follows ComfyUI Save Image behavior, including output
subfolders, counters, `%batch_num%`, and width/height/date tokens supported by
your ComfyUI version.

## Options

`empty_pass_behavior` controls what `passed_images` returns when every input
image fails:

```text
return_blank
return_best
pass_original_with_warning
return_empty
error
```

`return_blank` is the default because many ComfyUI image nodes crash on empty
image batches. Use `Facegate Save Image` for the actual discard/save
workflow.

`face_policy` controls which detected face is used:

```text
largest_face
center_face
any_face_passes
all_faces_must_pass
reject_multiple_faces
```

`match_policy` controls how references are evaluated:

```text
best_reference
average_reference_embedding
majority_vote
require_minimum_references
```

`provider`, `det_size`, and `model_root` control InsightFace runtime loading.
Enable `debug_preview` to draw face boxes and distance labels into
`debug_images`.
