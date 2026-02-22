import base64
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Legacy single-process options/functions are intentionally kept for future reuse,
# but are no longer exposed from the current app UI.
PROCESS_OPTIONS = [
    ("original", "原画像"),
    ("grayscale", "グレースケール"),
    ("gaussian_blur", "ガウシアンぼかし"),
    ("threshold", "二値化（固定閾値）"),
    ("adaptive_threshold", "二値化（適応的）"),
    ("canny", "Canny エッジ検出"),
    ("hough_lines", "Hough 線検出（確率的）"),
]
PROCESS_LABELS = dict(PROCESS_OPTIONS)

PIPELINE_STAGE_COUNT = 4
PIPELINE_OPERATION_OPTIONS = [
    ("brightness", "明るさ調整"),
    ("contrast", "コントラスト調整"),
    ("gamma", "ガンマ変換"),
    ("hue_shift", "色相シフト"),
    ("saturation", "彩度調整"),
    ("gaussian_blur", "ガウシアンぼかし"),
    ("grayscale", "グレースケール"),
    ("invert", "反転"),
]
PIPELINE_OPERATION_LABELS = dict(PIPELINE_OPERATION_OPTIONS)
PIPELINE_PARAM_USAGE = {
    "brightness": {"brightness"},
    "contrast": {"contrast"},
    "gamma": {"gamma"},
    "hue_shift": {"hue_deg"},
    "saturation": {"saturation_percent"},
    "gaussian_blur": {"blur_kernel"},
    "grayscale": set(),
    "invert": set(),
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

LAST_UPLOADED_IMAGE_BYTES: bytes | None = None
LAST_UPLOADED_FILENAME: str | None = None


def default_form_values() -> dict[str, int | str]:
    return {
        "process_type": "grayscale",
        "blur_kernel": 5,
        "threshold_value": 128,
        "adaptive_block_size": 11,
        "adaptive_c": 2,
        "canny_low": 80,
        "canny_high": 160,
        "hough_threshold": 50,
        "hough_min_line_length": 40,
        "hough_max_line_gap": 10,
    }


def default_pipeline_stage_values(stage_index: int) -> dict[str, int | float | str | bool]:
    default_operation = "brightness"
    default_enabled = False
    if stage_index == 1:
        default_operation = "contrast"
        default_enabled = True
    elif stage_index == 2:
        default_operation = "gamma"
        default_enabled = True

    return {
        "index": stage_index,
        "prefix": f"stage{stage_index}",
        "enabled": default_enabled,
        "operation": default_operation,
        "brightness": 0,
        "contrast": 1.2 if stage_index == 1 else 1.0,
        "gamma": 1.0,
        "hue_deg": 0,
        "saturation_percent": 100,
        "blur_kernel": 5,
    }


def default_pipeline_stages() -> list[dict[str, int | float | str | bool]]:
    return [default_pipeline_stage_values(i) for i in range(1, PIPELINE_STAGE_COUNT + 1)]


def parse_int(form, key: str, default: int, *, min_value: int, max_value: int) -> int:
    raw = form.get(key, "")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def parse_float(form, key: str, default: float, *, min_value: float, max_value: float) -> float:
    raw = form.get(key, "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def make_odd(value: int, *, minimum: int) -> int:
    if value < minimum:
        value = minimum
    if value % 2 == 0:
        value += 1
    return value


def decode_image_bytes(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("画像データが空です。")
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("画像として読み込めないファイルです。")
    return image


def decode_uploaded_image(file_storage) -> np.ndarray:
    return decode_image_bytes(file_storage.read())


def encode_image_data_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("画像の表示用エンコードに失敗しました。")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


# Legacy single-process executor (kept for later reintroduction)
def apply_process(image: np.ndarray, params: dict[str, int | str]) -> tuple[np.ndarray, dict[str, str]]:
    process_type = str(params["process_type"])
    info: dict[str, str] = {}

    if process_type == "original":
        return image.copy(), info
    if process_type == "grayscale":
        return to_gray(image), info
    if process_type == "gaussian_blur":
        kernel = make_odd(int(params["blur_kernel"]), minimum=1)
        info["blur_kernel"] = str(kernel)
        return cv2.GaussianBlur(image, (kernel, kernel), 0), info

    gray = to_gray(image)
    if process_type == "threshold":
        threshold_value = int(params["threshold_value"])
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        info["threshold_value"] = str(threshold_value)
        return binary, info
    if process_type == "adaptive_threshold":
        block_size = make_odd(int(params["adaptive_block_size"]), minimum=3)
        c_value = int(params["adaptive_c"])
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_value,
        )
        info["adaptive_block_size"] = str(block_size)
        info["adaptive_c"] = str(c_value)
        return adaptive, info
    if process_type == "canny":
        low = int(params["canny_low"])
        high = max(low + 1, int(params["canny_high"]))
        edges = cv2.Canny(gray, low, high)
        info["canny_low"] = str(low)
        info["canny_high"] = str(high)
        return edges, info
    if process_type == "hough_lines":
        low = int(params["canny_low"])
        high = max(low + 1, int(params["canny_high"]))
        hough_threshold = int(params["hough_threshold"])
        min_length = int(params["hough_min_line_length"])
        max_gap = int(params["hough_max_line_gap"])
        edges = cv2.Canny(gray, low, high)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_length,
            maxLineGap=max_gap,
        )
        overlay = image.copy()
        line_count = 0
        if lines is not None:
            line_count = len(lines)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(overlay, (x1, y1), (x2, y2), (20, 220, 20), 2)
        info["canny_low"] = str(low)
        info["canny_high"] = str(high)
        info["hough_threshold"] = str(hough_threshold)
        info["hough_min_line_length"] = str(min_length)
        info["hough_max_line_gap"] = str(max_gap)
        info["detected_lines"] = str(line_count)
        return overlay, info
    raise ValueError("未対応の処理タイプです。")


def parse_pipeline_stages(form) -> list[dict[str, int | float | str | bool]]:
    stages = default_pipeline_stages()
    valid_ops = set(PIPELINE_OPERATION_LABELS)

    for stage in stages:
        prefix = str(stage["prefix"])
        stage["enabled"] = f"{prefix}_enabled" in form
        operation = form.get(f"{prefix}_operation", str(stage["operation"]))
        if operation not in valid_ops:
            operation = "brightness"
        stage["operation"] = operation
        stage["brightness"] = parse_int(form, f"{prefix}_brightness", int(stage["brightness"]), min_value=-255, max_value=255)
        stage["contrast"] = parse_float(form, f"{prefix}_contrast", float(stage["contrast"]), min_value=0.1, max_value=4.0)
        stage["gamma"] = parse_float(form, f"{prefix}_gamma", float(stage["gamma"]), min_value=0.1, max_value=5.0)
        stage["hue_deg"] = parse_int(form, f"{prefix}_hue_deg", int(stage["hue_deg"]), min_value=-180, max_value=180)
        stage["saturation_percent"] = parse_int(form, f"{prefix}_saturation_percent", int(stage["saturation_percent"]), min_value=0, max_value=300)
        stage["blur_kernel"] = make_odd(parse_int(form, f"{prefix}_blur_kernel", int(stage["blur_kernel"]), min_value=1, max_value=99), minimum=1)
    return stages


def gamma_correct(image: np.ndarray, gamma_value: float) -> np.ndarray:
    table = np.array([np.clip(((i / 255.0) ** gamma_value) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def apply_pipeline_stage(image: np.ndarray, stage: dict[str, int | float | str | bool]) -> tuple[np.ndarray, dict[str, str]]:
    operation = str(stage["operation"])
    info: dict[str, str] = {}

    if operation == "brightness":
        delta = int(stage["brightness"])
        result = np.clip(image.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        info["brightness"] = str(delta)
        return result, info
    if operation == "contrast":
        alpha = float(stage["contrast"])
        result = np.clip((image.astype(np.float32) - 127.5) * alpha + 127.5, 0, 255).astype(np.uint8)
        info["contrast"] = f"{alpha:.2f}"
        return result, info
    if operation == "gamma":
        gamma_value = float(stage["gamma"])
        info["gamma"] = f"{gamma_value:.2f}"
        return gamma_correct(image, gamma_value), info
    if operation == "hue_shift":
        hue_deg = int(stage["hue_deg"])
        hsv = cv2.cvtColor(to_bgr(image), cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + int(round(hue_deg / 2))) % 180
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        info["hue_deg"] = str(hue_deg)
        return result, info
    if operation == "saturation":
        saturation_percent = int(stage["saturation_percent"])
        hsv = cv2.cvtColor(to_bgr(image), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * (saturation_percent / 100.0), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        info["saturation_percent"] = str(saturation_percent)
        return result, info
    if operation == "gaussian_blur":
        kernel = make_odd(int(stage["blur_kernel"]), minimum=1)
        info["blur_kernel"] = str(kernel)
        return cv2.GaussianBlur(image, (kernel, kernel), 0), info
    if operation == "grayscale":
        return to_gray(image), info
    if operation == "invert":
        return 255 - image, info
    raise ValueError("未対応のパイプライン処理です。")


def summarize_image(image: np.ndarray) -> dict[str, str]:
    return {
        "size": f"{image.shape[1]} x {image.shape[0]}",
        "channels": str(image.shape[2]) if image.ndim == 3 else "1",
        "dtype": str(image.dtype),
    }


def run_pipeline(original: np.ndarray, stages: list[dict[str, int | float | str | bool]]) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    current = original.copy()
    executed_steps: list[dict[str, object]] = []
    stage_previews: list[dict[str, object]] = []
    for stage in stages:
        if not bool(stage["enabled"]):
            continue
        current, info = apply_pipeline_stage(current, stage)
        operation = str(stage["operation"])
        executed_steps.append({
            "index": int(stage["index"]),
            "label": PIPELINE_OPERATION_LABELS[operation],
            "operation": operation,
            "info": info,
        })
        stage_previews.append({
            "index": int(stage["index"]),
            "label": PIPELINE_OPERATION_LABELS[operation],
            "src": encode_image_data_url(current),
            "image_info": summarize_image(current),
        })
    return current, executed_steps, stage_previews


def build_template_context(**kwargs):
    context = {
        "pipeline_operation_options": PIPELINE_OPERATION_OPTIONS,
        "pipeline_stages": default_pipeline_stages(),
        "pipeline_param_usage": {k: sorted(v) for k, v in PIPELINE_PARAM_USAGE.items()},
        "error": None,
        "result": None,
        "has_cached_image": LAST_UPLOADED_IMAGE_BYTES is not None,
        "cached_filename": LAST_UPLOADED_FILENAME,
    }
    context.update(kwargs)
    return context


@app.get("/")
def index():
    return render_template("index.html", **build_template_context())


@app.post("/process")
def process_image():
    global LAST_UPLOADED_IMAGE_BYTES, LAST_UPLOADED_FILENAME

    pipeline_stages = parse_pipeline_stages(request.form)
    file = request.files.get("image")
    has_new_upload = file is not None and file.filename != ""

    if not has_new_upload and LAST_UPLOADED_IMAGE_BYTES is None:
        return (
            render_template(
                "index.html",
                **build_template_context(pipeline_stages=pipeline_stages, error="画像ファイルを選択してください。"),
            ),
            400,
        )

    try:
        if has_new_upload:
            uploaded_bytes = file.read()
            original = decode_image_bytes(uploaded_bytes)
            LAST_UPLOADED_IMAGE_BYTES = uploaded_bytes
            LAST_UPLOADED_FILENAME = file.filename
            filename = file.filename
        else:
            original = decode_image_bytes(LAST_UPLOADED_IMAGE_BYTES or b"")
            filename = LAST_UPLOADED_FILENAME or "cached_image"

        final_image, executed_steps, stage_previews = run_pipeline(original, pipeline_stages)
        is_landscape = original.shape[1] >= original.shape[0]
        result = {
            "filename": filename,
            "used_cached_image": not has_new_upload,
            "original_src": encode_image_data_url(original),
            "final_src": encode_image_data_url(final_image),
            "executed_steps": executed_steps,
            "stage_previews": stage_previews,
            "original_info": summarize_image(original),
            "final_info": summarize_image(final_image),
            "preview_layout_class": "stack-vertical" if is_landscape else "stack-horizontal",
            "stage_layout_class": "stack-vertical" if is_landscape else "stack-horizontal",
        }
    except ValueError as exc:
        return (
            render_template(
                "index.html",
                **build_template_context(pipeline_stages=pipeline_stages, error=str(exc)),
            ),
            400,
        )

    return render_template("index.html", **build_template_context(pipeline_stages=pipeline_stages, result=result))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
