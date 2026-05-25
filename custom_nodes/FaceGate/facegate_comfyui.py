from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo


NODE_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

EMPTY_PASS_BEHAVIORS = (
    "return_blank",
    "return_best",
    "pass_original_with_warning",
    "return_empty",
    "error",
)
FACE_POLICIES = (
    "largest_face",
    "center_face",
    "any_face_passes",
    "all_faces_must_pass",
    "reject_multiple_faces",
)
MATCH_POLICIES = (
    "best_reference",
    "average_reference_embedding",
    "majority_vote",
    "require_minimum_references",
)
PROVIDER_MODES = ("auto", "cuda", "cpu")

_DLL_DIRECTORY_HANDLES: list[Any] = []
_DLL_DIRECTORY_PATHS: set[str] = set()
_APP_CACHE: dict[tuple[str, int, str], Any] = {}
_REFERENCE_CACHE: dict[tuple[tuple[str, int, str], str, tuple[tuple[str, int, int], ...]], "ReferenceSet"] = {}


@dataclass
class DetectedFace:
    bbox: tuple[float, float, float, float]
    embedding: np.ndarray


@dataclass
class FaceMatch:
    passed: bool
    distance: float
    similarity: float
    best_reference: str
    reference_matches: int
    required_reference_matches: int
    policy: str


@dataclass
class FaceEvaluation:
    bbox: tuple[float, float, float, float]
    selected: bool
    passed: bool
    distance: float | None
    similarity: float | None
    best_reference: str | None
    reference_matches: int | None
    required_reference_matches: int | None


@dataclass
class ImageGateResult:
    index: int
    passed: bool
    reason: str
    distance: float | None
    similarity: float | None
    best_reference: str | None
    face_count: int
    reference_matches: int | None
    required_reference_matches: int | None
    face_evaluations: list[FaceEvaluation]


@dataclass
class ReferenceSet:
    folder: Path
    references: list[tuple[Path, np.ndarray]]
    average_embedding: np.ndarray
    skipped_no_face: list[Path]
    signature: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class ModelLocation:
    model_dir: Path
    insightface_root: Path | None


def configure_windows_cuda_dll_paths() -> None:
    """Add user-provided CUDA DLL folders before onnxruntime is imported."""
    if os.name != "nt":
        return

    env_dirs = os.environ.get("FACE_COMPARE_CUDA_DLL_DIRS", "")
    cuda_dll_dirs = [Path(part.strip('"')) for part in env_dirs.split(os.pathsep) if part]

    for dll_dir in cuda_dll_dirs:
        if not dll_dir.is_dir():
            continue
        resolved = str(dll_dir.resolve())
        if resolved in _DLL_DIRECTORY_PATHS:
            continue
        _DLL_DIRECTORY_PATHS.add(resolved)
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(dll_dir)))
        os.environ["PATH"] = f"{dll_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def iter_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.glob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_reference_folder(reference_folder: str) -> Path:
    raw = (reference_folder or "").strip().strip('"')
    if not raw:
        raise ValueError("Set reference_folder to a folder that contains reference face images.")

    folder = Path(raw).expanduser()
    if not folder.is_absolute():
        folder = Path.cwd() / folder
    return folder.resolve()


def reference_signature(folder: Path) -> tuple[tuple[str, int, int], ...]:
    if not folder.is_dir():
        return tuple()

    signature = []
    for path in iter_images(folder):
        stat = path.stat()
        signature.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def image_to_rgb_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding.astype(np.float32)
    return (embedding / norm).astype(np.float32)


def cosine_distance(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    similarity = float(
        np.dot(reference, candidate) / (np.linalg.norm(reference) * np.linalg.norm(candidate))
    )
    return float(1.0 - similarity), similarity


def candidate_model_roots(model_root: str) -> list[Path]:
    candidates: list[Path] = []

    raw_model_root = (model_root or "").strip().strip('"')
    if raw_model_root:
        candidates.append(Path(raw_model_root).expanduser())

    try:
        import folder_paths  # type: ignore

        models_dir = Path(folder_paths.models_dir)
        try:
            candidates.extend(Path(path) for path in folder_paths.get_folder_paths("insightface"))
        except Exception:
            pass
        candidates.append(models_dir / "insightface")
        candidates.append(models_dir / "insightface" / "models")
        candidates.append(models_dir / "insightface" / "models" / "antelopev2")
        candidates.append(models_dir / "insightface" / "antelopev2")
        candidates.append(models_dir)
    except Exception:
        pass

    return candidates


def has_onnx_files(folder: Path) -> bool:
    return folder.is_dir() and any(folder.glob("*.onnx"))


def normalize_model_location(candidate: Path) -> ModelLocation | None:
    candidate = candidate.resolve()

    nested_model_dir = candidate / "models" / "antelopev2"
    if has_onnx_files(nested_model_dir):
        return ModelLocation(model_dir=nested_model_dir, insightface_root=candidate)

    if candidate.name.lower() == "models" and has_onnx_files(candidate / "antelopev2"):
        return ModelLocation(model_dir=candidate / "antelopev2", insightface_root=candidate.parent)

    if (
        candidate.name.lower() == "antelopev2"
        and candidate.parent.name.lower() == "models"
        and has_onnx_files(candidate)
    ):
        return ModelLocation(model_dir=candidate, insightface_root=candidate.parent.parent)

    if has_onnx_files(candidate / "antelopev2"):
        return ModelLocation(model_dir=candidate / "antelopev2", insightface_root=None)

    if candidate.name.lower() == "antelopev2" and has_onnx_files(candidate):
        return ModelLocation(model_dir=candidate, insightface_root=None)

    return None


def resolve_model_location(model_root: str) -> ModelLocation:
    checked: list[Path] = []
    for candidate in candidate_model_roots(model_root):
        checked.append(candidate)
        try:
            resolved = normalize_model_location(candidate)
        except OSError:
            resolved = None
        if resolved is not None:
            return resolved

    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "Could not find InsightFace antelopev2 models in ComfyUI's model paths. Checked:\n"
        f"{checked_text}\n\nExpected layout: <model-root>/models/antelopev2/*.onnx"
        "\nor: <ComfyUI>/models/insightface/antelopev2/*.onnx"
    )


def provider_list(provider: str) -> list[str]:
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if provider == "cuda":
        return ["CUDAExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def build_face_app_from_direct_model_dir(model_dir: Path, providers: list[str]) -> Any:
    import glob
    import os.path as osp

    import onnxruntime
    from insightface import model_zoo
    from insightface.app.face_analysis import FaceAnalysis

    onnxruntime.set_default_logger_severity(3)

    app = FaceAnalysis.__new__(FaceAnalysis)
    app.models = {}
    app.model_dir = str(model_dir)

    onnx_files = sorted(glob.glob(osp.join(str(model_dir), "*.onnx")))
    for onnx_file in onnx_files:
        model = model_zoo.get_model(onnx_file, providers=providers)
        if model is None:
            continue
        if model.taskname not in app.models:
            app.models[model.taskname] = model

    if "detection" not in app.models:
        raise RuntimeError(f"InsightFace detection model not found in: {model_dir}")

    app.det_model = app.models["detection"]
    return app


def get_face_app(model_root: str, det_size: int, provider: str) -> tuple[Any, tuple[str, int, str], Path]:
    model_location = resolve_model_location(model_root)
    cache_key = (str(model_location.model_dir), int(det_size), provider)

    cached = _APP_CACHE.get(cache_key)
    if cached is not None:
        return cached, cache_key, model_location.model_dir

    configure_windows_cuda_dll_paths()
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    providers = provider_list(provider)
    if model_location.insightface_root is not None:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="antelopev2",
            root=str(model_location.insightface_root),
            providers=providers,
        )
    else:
        app = build_face_app_from_direct_model_dir(model_location.model_dir, providers)

    app.prepare(
        ctx_id=-1 if provider == "cpu" else 0,
        det_size=(int(det_size), int(det_size)),
    )
    _APP_CACHE[cache_key] = app
    return app, cache_key, model_location.model_dir


def detect_faces(app: Any, image_rgb: np.ndarray) -> list[DetectedFace]:
    faces = app.get(image_rgb)
    detected: list[DetectedFace] = []

    for face in faces:
        embedding = getattr(face, "normed_embedding", None)
        bbox = getattr(face, "bbox", None)
        if embedding is None or bbox is None:
            continue
        detected.append(
            DetectedFace(
                bbox=tuple(float(value) for value in bbox),
                embedding=np.asarray(embedding, dtype=np.float32),
            )
        )

    return detected


def largest_face(faces: list[DetectedFace]) -> DetectedFace:
    return max(
        faces,
        key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
    )


def center_face(faces: list[DetectedFace], width: int, height: int) -> DetectedFace:
    center_x = width / 2
    center_y = height / 2

    def squared_center_distance(face: DetectedFace) -> float:
        x1, y1, x2, y2 = face.bbox
        face_x = (x1 + x2) / 2
        face_y = (y1 + y2) / 2
        return (face_x - center_x) ** 2 + (face_y - center_y) ** 2

    return min(faces, key=squared_center_distance)


def load_reference_set(
    app: Any,
    app_key: tuple[str, int, str],
    reference_folder: str,
) -> ReferenceSet:
    folder = resolve_reference_folder(reference_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Reference folder not found: {folder}")

    signature = reference_signature(folder)
    if not signature:
        raise RuntimeError(f"No reference images found in: {folder}")

    cache_key = (app_key, str(folder), signature)
    cached = _REFERENCE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    references: list[tuple[Path, np.ndarray]] = []
    skipped_no_face: list[Path] = []

    for path in iter_images(folder):
        faces = detect_faces(app, image_to_rgb_array(path))
        if not faces:
            skipped_no_face.append(path)
            continue
        references.append((path, largest_face(faces).embedding))

    if not references:
        raise RuntimeError(f"No usable faces found in reference images: {folder}")

    average_embedding = normalize_embedding(np.mean([embedding for _, embedding in references], axis=0))
    reference_set = ReferenceSet(
        folder=folder,
        references=references,
        average_embedding=average_embedding,
        skipped_no_face=skipped_no_face,
        signature=signature,
    )
    _REFERENCE_CACHE[cache_key] = reference_set
    return reference_set


def match_face(
    reference_set: ReferenceSet,
    candidate_embedding: np.ndarray,
    threshold: float,
    match_policy: str,
    minimum_reference_matches: int,
) -> FaceMatch:
    distances: list[tuple[Path, float, float]] = []
    for reference_path, reference_embedding in reference_set.references:
        distance, similarity = cosine_distance(reference_embedding, candidate_embedding)
        distances.append((reference_path, distance, similarity))

    best_reference, best_distance, best_similarity = min(distances, key=lambda item: item[1])
    reference_matches = sum(1 for _, distance, _ in distances if distance <= threshold)

    if match_policy == "average_reference_embedding":
        average_distance, average_similarity = cosine_distance(
            reference_set.average_embedding,
            candidate_embedding,
        )
        return FaceMatch(
            passed=average_distance <= threshold,
            distance=average_distance,
            similarity=average_similarity,
            best_reference="average_reference_embedding",
            reference_matches=reference_matches,
            required_reference_matches=1,
            policy=match_policy,
        )

    if match_policy == "majority_vote":
        required = (len(reference_set.references) // 2) + 1
        return FaceMatch(
            passed=reference_matches >= required,
            distance=best_distance,
            similarity=best_similarity,
            best_reference=str(best_reference),
            reference_matches=reference_matches,
            required_reference_matches=required,
            policy=match_policy,
        )

    if match_policy == "require_minimum_references":
        required = max(1, min(int(minimum_reference_matches), len(reference_set.references)))
        return FaceMatch(
            passed=reference_matches >= required,
            distance=best_distance,
            similarity=best_similarity,
            best_reference=str(best_reference),
            reference_matches=reference_matches,
            required_reference_matches=required,
            policy=match_policy,
        )

    return FaceMatch(
        passed=best_distance <= threshold,
        distance=best_distance,
        similarity=best_similarity,
        best_reference=str(best_reference),
        reference_matches=reference_matches,
        required_reference_matches=1,
        policy="best_reference",
    )


def face_evaluation(face: DetectedFace, face_match: FaceMatch, selected: bool) -> FaceEvaluation:
    return FaceEvaluation(
        bbox=face.bbox,
        selected=selected,
        passed=face_match.passed,
        distance=face_match.distance,
        similarity=face_match.similarity,
        best_reference=face_match.best_reference,
        reference_matches=face_match.reference_matches,
        required_reference_matches=face_match.required_reference_matches,
    )


def evaluate_image(
    app: Any,
    reference_set: ReferenceSet,
    image_rgb: np.ndarray,
    index: int,
    threshold: float,
    face_policy: str,
    match_policy: str,
    minimum_reference_matches: int,
) -> ImageGateResult:
    height, width = image_rgb.shape[:2]
    faces = detect_faces(app, image_rgb)

    if not faces:
        return ImageGateResult(
            index=index,
            passed=False,
            reason="no_face",
            distance=None,
            similarity=None,
            best_reference=None,
            face_count=0,
            reference_matches=None,
            required_reference_matches=None,
            face_evaluations=[],
        )

    face_matches = [
        match_face(reference_set, face.embedding, threshold, match_policy, minimum_reference_matches)
        for face in faces
    ]

    if face_policy == "reject_multiple_faces" and len(faces) > 1:
        evaluations = [
            face_evaluation(face, face_match, selected=False)
            for face, face_match in zip(faces, face_matches)
        ]
        best = min(face_matches, key=lambda item: item.distance)
        return ImageGateResult(
            index=index,
            passed=False,
            reason="multiple_faces",
            distance=best.distance,
            similarity=best.similarity,
            best_reference=best.best_reference,
            face_count=len(faces),
            reference_matches=best.reference_matches,
            required_reference_matches=best.required_reference_matches,
            face_evaluations=evaluations,
        )

    if face_policy == "any_face_passes":
        selected_match = min(
            (face_match for face_match in face_matches if face_match.passed),
            key=lambda item: item.distance,
            default=min(face_matches, key=lambda item: item.distance),
        )
        evaluations = [
            face_evaluation(face, face_match, selected=face_match is selected_match)
            for face, face_match in zip(faces, face_matches)
        ]
        passed = any(face_match.passed for face_match in face_matches)
        return ImageGateResult(
            index=index,
            passed=passed,
            reason="ok" if passed else "above_threshold",
            distance=selected_match.distance,
            similarity=selected_match.similarity,
            best_reference=selected_match.best_reference,
            face_count=len(faces),
            reference_matches=selected_match.reference_matches,
            required_reference_matches=selected_match.required_reference_matches,
            face_evaluations=evaluations,
        )

    if face_policy == "all_faces_must_pass":
        passed = all(face_match.passed for face_match in face_matches)
        selected_match = max(face_matches, key=lambda item: item.distance)
        evaluations = [
            face_evaluation(face, face_match, selected=face_match is selected_match)
            for face, face_match in zip(faces, face_matches)
        ]
        return ImageGateResult(
            index=index,
            passed=passed,
            reason="ok" if passed else "above_threshold",
            distance=selected_match.distance,
            similarity=selected_match.similarity,
            best_reference=selected_match.best_reference,
            face_count=len(faces),
            reference_matches=selected_match.reference_matches,
            required_reference_matches=selected_match.required_reference_matches,
            face_evaluations=evaluations,
        )

    selected_face = center_face(faces, width, height) if face_policy == "center_face" else largest_face(faces)
    selected_index = faces.index(selected_face)
    selected_match = face_matches[selected_index]
    evaluations = [
        face_evaluation(face, face_match, selected=face_index == selected_index)
        for face_index, (face, face_match) in enumerate(zip(faces, face_matches))
    ]

    return ImageGateResult(
        index=index,
        passed=selected_match.passed,
        reason="ok" if selected_match.passed else "above_threshold",
        distance=selected_match.distance,
        similarity=selected_match.similarity,
        best_reference=selected_match.best_reference,
        face_count=len(faces),
        reference_matches=selected_match.reference_matches,
        required_reference_matches=selected_match.required_reference_matches,
        face_evaluations=evaluations,
    )


def tensor_image_to_rgb(image: Any) -> np.ndarray:
    array = image.detach().cpu().numpy()
    array = np.clip(array, 0.0, 1.0)

    if array.ndim != 3:
        raise ValueError(f"Expected IMAGE item with shape HWC, got {array.shape}")

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] > 3:
        array = array[:, :, :3]

    return (array * 255.0 + 0.5).astype(np.uint8)


def rgb_arrays_to_tensor(arrays: list[np.ndarray], like_images: Any) -> Any:
    import torch

    batch = np.stack([np.clip(array, 0, 255).astype(np.float32) / 255.0 for array in arrays], axis=0)
    tensor = torch.from_numpy(batch)
    return tensor.to(device=like_images.device, dtype=like_images.dtype)


def select_images(images: Any, indices: list[int]) -> Any:
    import torch

    index_tensor = torch.tensor(indices, device=images.device, dtype=torch.long)
    return images.index_select(0, index_tensor)


def blank_image_like(images: Any) -> Any:
    import torch

    _, height, width, channels = images.shape
    return torch.zeros((1, height, width, channels), device=images.device, dtype=images.dtype)


def empty_image_batch_like(images: Any) -> Any:
    return images[:0]


def output_passed_images(
    images: Any,
    pass_indices: list[int],
    results: list[ImageGateResult],
    empty_pass_behavior: str,
    warnings: list[str],
) -> Any:
    if pass_indices:
        return select_images(images, pass_indices)

    if empty_pass_behavior == "error":
        raise RuntimeError("FaceGate rejected every image.")

    if empty_pass_behavior == "return_empty":
        warnings.append("No images passed; passed_images is an empty batch.")
        return empty_image_batch_like(images)

    if empty_pass_behavior == "return_blank":
        warnings.append("No images passed; passed_images contains a blank placeholder.")
        return blank_image_like(images)

    if empty_pass_behavior == "pass_original_with_warning":
        warnings.append("No images passed; passed_images contains the original input batch.")
        return images

    best_index = best_candidate_index(results)
    warnings.append(f"No images passed; passed_images contains best candidate index {best_index}.")
    return select_images(images, [best_index])


def output_failed_images(images: Any, fail_indices: list[int], warnings: list[str]) -> Any:
    if fail_indices:
        return select_images(images, fail_indices)
    warnings.append("No images failed; failed_images is an empty batch.")
    return empty_image_batch_like(images)


def best_candidate_index(results: list[ImageGateResult]) -> int:
    with_scores = [result for result in results if result.distance is not None]
    if with_scores:
        return min(
            with_scores,
            key=lambda result: result.distance if result.distance is not None else float("inf"),
        ).index
    return results[0].index if results else 0


def build_pass_mask(images: Any, results: list[ImageGateResult]) -> Any:
    import torch

    _, height, width, _ = images.shape
    values = torch.tensor(
        [1.0 if result.passed else 0.0 for result in results],
        device=images.device,
        dtype=torch.float32,
    )
    return values[:, None, None].expand(len(results), height, width).clone()


def build_pass_mask_image(images: Any, results: list[ImageGateResult]) -> Any:
    mask = build_pass_mask(images, results)
    return mask[:, :, :, None].expand(-1, -1, -1, 3).to(dtype=images.dtype)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str, color: tuple[int, int, int]) -> None:
    x, y = xy
    text_bbox = draw.textbbox((x, y), label)
    padding = 3
    background = (
        text_bbox[0] - padding,
        text_bbox[1] - padding,
        text_bbox[2] + padding,
        text_bbox[3] + padding,
    )
    draw.rectangle(background, fill=(0, 0, 0))
    draw.text((x, y), label, fill=color)


def debug_image(image_rgb: np.ndarray, result: ImageGateResult, enabled: bool) -> np.ndarray:
    if not enabled:
        return image_rgb

    image = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(image)
    line_width = max(2, min(image.size) // 256)

    if not result.face_evaluations:
        draw_label(draw, (8, 8), result.reason, (255, 80, 80))
        return np.asarray(image)

    for face in result.face_evaluations:
        x1, y1, x2, y2 = [int(round(value)) for value in face.bbox]
        color = (80, 220, 120) if face.passed else (255, 80, 80)
        if face.selected:
            color = (80, 170, 255) if face.passed else (255, 190, 80)

        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        if face.distance is None:
            label = "no_score"
        else:
            status = "pass" if face.passed else "fail"
            selected = "*" if face.selected else ""
            label = f"{selected}{status} d={face.distance:.4f}"
        draw_label(draw, (max(0, x1), max(0, y1 - 16)), label, color)

    draw_label(
        draw,
        (8, 8),
        f"image {result.index}: {'pass' if result.passed else 'fail'} {result.reason}",
        (80, 220, 120) if result.passed else (255, 80, 80),
    )
    return np.asarray(image)


def float_or_none(value: float | None) -> float | None:
    return None if value is None else float(value)


def result_to_dict(result: ImageGateResult) -> dict[str, Any]:
    return {
        "index": result.index,
        "passed": result.passed,
        "reason": result.reason,
        "cosine_distance": float_or_none(result.distance),
        "cosine_similarity": float_or_none(result.similarity),
        "best_reference": result.best_reference,
        "face_count": result.face_count,
        "reference_matches": result.reference_matches,
        "required_reference_matches": result.required_reference_matches,
        "faces": [
            {
                "bbox": [float(value) for value in face.bbox],
                "selected": face.selected,
                "passed": face.passed,
                "cosine_distance": float_or_none(face.distance),
                "cosine_similarity": float_or_none(face.similarity),
                "best_reference": face.best_reference,
                "reference_matches": face.reference_matches,
                "required_reference_matches": face.required_reference_matches,
            }
            for face in result.face_evaluations
        ],
    }


def build_scores_json(
    reference_set: ReferenceSet,
    model_path: Path,
    threshold: float,
    face_policy: str,
    match_policy: str,
    minimum_reference_matches: int,
    provider: str,
    det_size: int,
    empty_pass_behavior: str,
    results: list[ImageGateResult],
    warnings: list[str],
) -> str:
    payload = {
        "threshold": float(threshold),
        "face_policy": face_policy,
        "match_policy": match_policy,
        "minimum_reference_matches": int(minimum_reference_matches),
        "provider": provider,
        "det_size": int(det_size),
        "model_path": str(model_path),
        "reference_folder": str(reference_set.folder),
        "reference_count": len(reference_set.references),
        "skipped_reference_count": len(reference_set.skipped_no_face),
        "skipped_references": [str(path) for path in reference_set.skipped_no_face],
        "empty_pass_behavior": empty_pass_behavior,
        "passed_count": sum(1 for result in results if result.passed),
        "failed_count": sum(1 for result in results if not result.passed),
        "warnings": warnings,
        "results": [result_to_dict(result) for result in results],
    }
    return json.dumps(payload, indent=2)


def parse_scores(scores: str) -> dict[str, Any]:
    if not scores:
        return {}
    try:
        payload = json.loads(scores)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def passed_indices_from_scores(scores: str, image_count: int) -> list[int]:
    payload = parse_scores(scores)
    passed_count = int(payload.get("passed_count") or 0)
    if passed_count <= 0:
        return []

    results = payload.get("results")
    if isinstance(results, list) and len(results) == image_count:
        indices: list[int] = []
        for result in results:
            if not isinstance(result, dict) or not result.get("passed"):
                continue
            index = result.get("index")
            if isinstance(index, int) and 0 <= index < image_count:
                indices.append(index)
        return indices

    if image_count == passed_count:
        return list(range(image_count))

    return list(range(min(image_count, passed_count)))


def image_tensor_to_pil(image: Any) -> Image.Image:
    array = 255.0 * image.detach().cpu().numpy()
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def comfy_date_format_to_strftime(date_format: str) -> str:
    replacements = (
        ("yyyy", "%Y"),
        ("YYYY", "%Y"),
        ("yy", "%y"),
        ("YY", "%y"),
        ("MM", "%m"),
        ("dd", "%d"),
        ("DD", "%d"),
        ("HH", "%H"),
        ("hh", "%H"),
        ("mm", "%M"),
        ("ss", "%S"),
    )
    strftime_format = date_format
    for source, replacement in replacements:
        strftime_format = strftime_format.replace(source, replacement)
    return strftime_format


def expand_comfy_filename_prefix(filename_prefix: str) -> str:
    def replace_date(match: re.Match[str]) -> str:
        date_format = comfy_date_format_to_strftime(match.group(1))
        return time.strftime(date_format)

    return re.sub(r"%date:([^%]+)%", replace_date, filename_prefix)


def png_metadata(prompt: Any, extra_pnginfo: Any) -> PngInfo | None:
    try:
        from comfy.cli_args import args  # type: ignore

        if args.disable_metadata:
            return None
    except Exception:
        pass

    metadata = PngInfo()
    if prompt is not None:
        metadata.add_text("prompt", json.dumps(prompt))
    if extra_pnginfo is not None:
        for key, value in extra_pnginfo.items():
            metadata.add_text(key, json.dumps(value))
    return metadata


class FaceGateSavePassedImages:
    CATEGORY = "FaceGate"
    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    DESCRIPTION = "Save only images that FaceGate marked as passed. Saves nothing when none passed."

    def __init__(self) -> None:
        import folder_paths  # type: ignore

        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "optional": {
                "scores": (
                    "STRING",
                    {
                        "forceInput": True,
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def save_images(
        self,
        images: Any,
        filename_prefix: str = "ComfyUI",
        scores: str = "",
        prompt: Any = None,
        extra_pnginfo: Any = None,
    ) -> dict[str, Any]:
        import folder_paths  # type: ignore

        filename_prefix += self.prefix_append
        filename_prefix = expand_comfy_filename_prefix(filename_prefix)
        selected_indices = (
            passed_indices_from_scores(scores, int(images.shape[0]))
            if scores
            else list(range(int(images.shape[0])))
        )
        if not selected_indices:
            return {
                "ui": {
                    "images": [],
                    "facegate": ["No images passed; skipped saving."],
                }
            }

        first = images[selected_indices[0]]
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            first.shape[1],
            first.shape[0],
        )
        metadata = png_metadata(prompt, extra_pnginfo)

        results = []
        for batch_number, image_index in enumerate(selected_indices):
            image = image_tensor_to_pil(images[image_index])
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            image.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append(
                {
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type,
                }
            )
            counter += 1

        return {"ui": {"images": results}}


class FaceGateNode:
    CATEGORY = "FaceGate"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "passed_images",
        "failed_images",
        "pass_mask",
        "scores",
        "debug_images",
        "pass_mask_image",
    )
    FUNCTION = "filter_images"
    DESCRIPTION = "Filter a ComfyUI IMAGE batch by InsightFace similarity to a reference folder."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "reference_folder": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.68,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                    },
                ),
            },
            "optional": {
                "empty_pass_behavior": (list(EMPTY_PASS_BEHAVIORS), {"default": "return_blank"}),
                "face_policy": (list(FACE_POLICIES), {"default": "largest_face"}),
                "match_policy": (list(MATCH_POLICIES), {"default": "best_reference"}),
                "minimum_reference_matches": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 999,
                        "step": 1,
                    },
                ),
                "model_root": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
                "det_size": (
                    "INT",
                    {
                        "default": 640,
                        "min": 128,
                        "max": 2048,
                        "step": 64,
                    },
                ),
                "provider": (list(PROVIDER_MODES), {"default": "auto"}),
                "debug_preview": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *args: Any, **kwargs: Any) -> str:
        reference_folder = kwargs.get("reference_folder")
        if reference_folder is None and len(args) >= 2:
            reference_folder = args[1]
        if reference_folder is None:
            reference_folder = ""
        try:
            folder = resolve_reference_folder(reference_folder)
            signature = reference_signature(folder)
            return json.dumps(
                {
                    "reference_folder": str(folder),
                    "reference_signature": signature,
                },
                sort_keys=True,
            )
        except Exception as exc:
            return f"reference_signature_error:{exc}"

    def filter_images(
        self,
        images: Any,
        reference_folder: str,
        threshold: float,
        empty_pass_behavior: str = "return_blank",
        face_policy: str = "largest_face",
        match_policy: str = "best_reference",
        minimum_reference_matches: int = 1,
        model_root: str = "",
        det_size: int = 640,
        provider: str = "auto",
        debug_preview: bool = False,
    ) -> tuple[Any, Any, Any, str, Any, Any]:
        if images is None or len(images.shape) != 4:
            raise ValueError("FaceGate expected images with shape BHWC.")
        if images.shape[0] == 0:
            raise ValueError("FaceGate received an empty image batch.")
        if empty_pass_behavior not in EMPTY_PASS_BEHAVIORS:
            raise ValueError(f"Unsupported empty_pass_behavior: {empty_pass_behavior}")
        if face_policy not in FACE_POLICIES:
            raise ValueError(f"Unsupported face_policy: {face_policy}")
        if match_policy not in MATCH_POLICIES:
            raise ValueError(f"Unsupported match_policy: {match_policy}")
        if provider not in PROVIDER_MODES:
            raise ValueError(f"Unsupported provider: {provider}")

        threshold = float(threshold)
        det_size = int(det_size)
        minimum_reference_matches = int(minimum_reference_matches)

        app, app_key, resolved_model_path = get_face_app(model_root, det_size, provider)
        reference_set = load_reference_set(app, app_key, reference_folder)

        rgb_images = [tensor_image_to_rgb(images[index]) for index in range(images.shape[0])]
        results = [
            evaluate_image(
                app=app,
                reference_set=reference_set,
                image_rgb=image_rgb,
                index=index,
                threshold=threshold,
                face_policy=face_policy,
                match_policy=match_policy,
                minimum_reference_matches=minimum_reference_matches,
            )
            for index, image_rgb in enumerate(rgb_images)
        ]

        pass_indices = [result.index for result in results if result.passed]
        fail_indices = [result.index for result in results if not result.passed]
        warnings: list[str] = []

        passed_images = output_passed_images(
            images,
            pass_indices,
            results,
            empty_pass_behavior,
            warnings,
        )
        failed_images = output_failed_images(images, fail_indices, warnings)
        pass_mask = build_pass_mask(images, results)
        pass_mask_image = build_pass_mask_image(images, results)
        debug_arrays = [
            debug_image(image_rgb, result, enabled=bool(debug_preview))
            for image_rgb, result in zip(rgb_images, results)
        ]
        debug_images = rgb_arrays_to_tensor(debug_arrays, images)
        scores = build_scores_json(
            reference_set=reference_set,
            model_path=resolved_model_path,
            threshold=threshold,
            face_policy=face_policy,
            match_policy=match_policy,
            minimum_reference_matches=minimum_reference_matches,
            provider=provider,
            det_size=det_size,
            empty_pass_behavior=empty_pass_behavior,
            results=results,
            warnings=warnings,
        )

        return passed_images, failed_images, pass_mask, scores, debug_images, pass_mask_image
