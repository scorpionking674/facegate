import argparse
import csv
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_INSIGHTFACE_ROOT = PROJECT_DIR / "models" / "insightface"
REFERENCE_DIR = PROJECT_DIR / "reference_image"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_DLL_DIRECTORY_HANDLES = []


def configure_windows_cuda_dll_paths() -> None:
    """Add user-provided CUDA DLL folders before onnxruntime is imported."""
    if os.name != "nt":
        return

    env_dirs = os.environ.get("FACE_COMPARE_CUDA_DLL_DIRS", "")
    cuda_dll_dirs = [Path(part.strip('"')) for part in env_dirs.split(os.pathsep) if part]

    for dll_dir in cuda_dll_dirs:
        if not dll_dir.is_dir():
            continue
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(dll_dir)))
        os.environ["PATH"] = f"{dll_dir}{os.pathsep}{os.environ.get('PATH', '')}"


configure_windows_cuda_dll_paths()
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from insightface.app import FaceAnalysis


@dataclass
class CompareResult:
    source: Path
    status: str
    cosine_distance: float | None
    cosine_similarity: float | None
    best_reference: Path | None
    face_count: int
    message: str
    output_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare reference images from the reference_image folder against "
            "every image in a folder using InsightFace antelopev2 and cosine distance."
        )
    )
    parser.add_argument("--input-folder", "-i", type=Path, help="Folder of images to compare.")
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.68,
        help="Cosine distance threshold. Lower is stricter; pass when distance <= threshold.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help=(
            "InsightFace root folder containing models\\antelopev2. "
            "Defaults to this project's models\\insightface folder."
        ),
    )
    parser.add_argument("--det-size", type=int, default=640, help="Detection input size.")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search input folder recursively. Enabled by default.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Clear passed_comparison and failed_comparison before running.",
    )
    return parser.parse_args()


def prompt_path(label: str, must_be_file: bool = False, must_be_dir: bool = False) -> Path:
    while True:
        raw = input(f"{label}: ").strip().strip('"')
        path = Path(raw)
        if must_be_file and not path.is_file():
            print(f"File not found: {path}")
            continue
        if must_be_dir and not path.is_dir():
            print(f"Folder not found: {path}")
            continue
        return path


def prompt_threshold(default: float) -> float:
    while True:
        raw = input(f"Cosine distance threshold [default {default}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("Enter a number, for example 0.68")
            continue
        if value < 0:
            print("Threshold must be 0 or higher.")
            continue
        return value


def resolve_model_root(model_root: Path | None) -> Path:
    candidates = []
    if model_root is not None:
        candidates.append(model_root)

    env_model_root = os.environ.get("FACE_COMPARE_MODEL_ROOT", "").strip().strip('"')
    if env_model_root:
        candidates.append(Path(env_model_root))

    candidates.append(LOCAL_INSIGHTFACE_ROOT)

    for candidate in candidates:
        if (candidate / "models" / "antelopev2").is_dir():
            return candidate

    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find InsightFace antelopev2 models. Checked:\n"
        f"{checked}\n\nExpected layout: <model-root>\\models\\antelopev2\\*.onnx"
    )


def load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def build_app(model_root: Path, det_size: int) -> FaceAnalysis:
    app = FaceAnalysis(
        name="antelopev2",
        root=str(model_root),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    return app


def get_largest_face_embedding(app: FaceAnalysis, image_path: Path) -> tuple[np.ndarray | None, int]:
    image = load_image_rgb(image_path)
    faces = app.get(image)
    if not faces:
        return None, 0

    largest = max(
        faces,
        key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
    )
    return np.asarray(largest.normed_embedding, dtype=np.float32), len(faces)


def cosine_distance(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    similarity = float(
        np.dot(reference, candidate) / (np.linalg.norm(reference) * np.linalg.norm(candidate))
    )
    return float(1.0 - similarity), similarity


def load_reference_embeddings(app: FaceAnalysis) -> list[tuple[Path, np.ndarray]]:
    reference_paths = iter_images(REFERENCE_DIR, recursive=False)

    references: list[tuple[Path, np.ndarray]] = []
    skipped = 0
    for reference_path in reference_paths:
        embedding, face_count = get_largest_face_embedding(app, reference_path)
        if embedding is None:
            skipped += 1
            print(f"Skipping reference with no detected face: {reference_path}")
            continue
        references.append((reference_path, embedding))
        print(f"Loaded reference faces={face_count}: {reference_path}")

    if not references:
        raise RuntimeError(
            f"No usable faces found in reference images. Check images in: {REFERENCE_DIR}"
        )

    if skipped:
        print(f"Skipped {skipped} reference image(s) with no detected face.")
    print(f"Using {len(references)} reference image(s). Decision mode: best match.")
    return references


def best_reference_match(
    references: list[tuple[Path, np.ndarray]], candidate: np.ndarray
) -> tuple[float, float, Path]:
    best_distance = float("inf")
    best_similarity = -float("inf")
    best_path = references[0][0]

    for reference_path, reference_embedding in references:
        distance, similarity = cosine_distance(reference_embedding, candidate)
        if distance < best_distance:
            best_distance = distance
            best_similarity = similarity
            best_path = reference_path

    return best_distance, best_similarity, best_path


def iter_images(folder: Path, recursive: bool) -> list[Path]:
    files = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(path for path in files if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def unique_destination(folder: Path, source: Path) -> Path:
    destination = folder / source.name
    if not destination.exists():
        return destination

    index = 1
    while True:
        candidate = folder / f"{source.stem}_{index}{source.suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def prepare_output_dirs(clear_output: bool) -> tuple[Path, Path]:
    passed_dir = PROJECT_DIR / "passed_comparison"
    failed_dir = PROJECT_DIR / "failed_comparison"

    if clear_output:
        for folder in (passed_dir, failed_dir):
            if folder.exists():
                shutil.rmtree(folder)

    passed_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    return passed_dir, failed_dir


def write_report(results: list[CompareResult]) -> Path:
    report_path = PROJECT_DIR / "comparison_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source",
                "status",
                "cosine_distance",
                "cosine_similarity",
                "best_reference",
                "face_count",
                "message",
                "output_path",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    str(result.source),
                    result.status,
                    "" if result.cosine_distance is None else f"{result.cosine_distance:.8f}",
                    "" if result.cosine_similarity is None else f"{result.cosine_similarity:.8f}",
                    "" if result.best_reference is None else str(result.best_reference),
                    result.face_count,
                    result.message,
                    "" if result.output_path is None else str(result.output_path),
                ]
            )
    return report_path


def compare_folder(
    app: FaceAnalysis,
    references: list[tuple[Path, np.ndarray]],
    input_folder: Path,
    threshold: float,
    recursive: bool,
    clear_output: bool,
) -> tuple[list[CompareResult], Path]:
    passed_dir, failed_dir = prepare_output_dirs(clear_output)

    results: list[CompareResult] = []
    reference_resolved = {reference_path.resolve() for reference_path, _ in references}

    for image_path in iter_images(input_folder, recursive):
        if image_path.resolve() in reference_resolved:
            continue

        try:
            candidate_embedding, face_count = get_largest_face_embedding(app, image_path)
            if candidate_embedding is None:
                destination = unique_destination(failed_dir, image_path)
                shutil.copy2(image_path, destination)
                results.append(
                    CompareResult(
                        source=image_path,
                        status="failed",
                        cosine_distance=None,
                        cosine_similarity=None,
                        best_reference=None,
                        face_count=0,
                        message="no face detected",
                        output_path=destination,
                    )
                )
                print(f"FAIL no-face: {image_path}")
                continue

            distance, similarity, best_reference = best_reference_match(references, candidate_embedding)
            passed = distance <= threshold
            destination = unique_destination(passed_dir if passed else failed_dir, image_path)
            shutil.copy2(image_path, destination)

            status = "passed" if passed else "failed"
            results.append(
                CompareResult(
                    source=image_path,
                    status=status,
                    cosine_distance=distance,
                    cosine_similarity=similarity,
                    best_reference=best_reference,
                    face_count=face_count,
                    message="ok",
                    output_path=destination,
                )
            )
            print(
                f"{status.upper()} distance={distance:.6f} "
                f"similarity={similarity:.6f} faces={face_count} "
                f"best_ref={best_reference.name}: {image_path}"
            )
        except Exception as exc:
            destination = unique_destination(failed_dir, image_path)
            shutil.copy2(image_path, destination)
            results.append(
                CompareResult(
                    source=image_path,
                    status="failed",
                    cosine_distance=None,
                    cosine_similarity=None,
                    best_reference=None,
                    face_count=0,
                    message=f"error: {exc}",
                    output_path=destination,
                )
            )
            print(f"FAIL error: {image_path} ({exc})")

    report_path = write_report(results)
    return results, report_path


def main() -> int:
    args = parse_args()

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    if not iter_images(REFERENCE_DIR, recursive=False):
        print(f"No reference images found. Add one or more images to: {REFERENCE_DIR}", file=sys.stderr)
        return 2

    input_folder = args.input_folder or prompt_path("Folder to compare", must_be_dir=True)
    threshold = args.threshold if args.input_folder else prompt_threshold(args.threshold)

    input_folder = input_folder.resolve()

    if not input_folder.is_dir():
        print(f"Input folder not found: {input_folder}", file=sys.stderr)
        return 2

    try:
        model_root = resolve_model_root(args.model_root)
        print(f"Using InsightFace model root: {model_root}")
        print("Using providers: CUDAExecutionProvider, CPUExecutionProvider fallback")
        app = build_app(model_root, args.det_size)
        references = load_reference_embeddings(app)
        results, report_path = compare_folder(
            app=app,
            references=references,
            input_folder=input_folder,
            threshold=threshold,
            recursive=args.recursive,
            clear_output=args.clear_output,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    passed = sum(1 for result in results if result.status == "passed")
    failed = sum(1 for result in results if result.status == "failed")
    print()
    print(f"Done. Passed: {passed}. Failed: {failed}.")
    print(f"Passed folder: {PROJECT_DIR / 'passed_comparison'}")
    print(f"Failed folder: {PROJECT_DIR / 'failed_comparison'}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
