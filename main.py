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
    ("threshold", "二値化"),
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
    "threshold": {"threshold_mode", "threshold_value", "adaptive_block_size", "adaptive_c"},
    "grayscale": set(),
    "invert": set(),
}

ANALYSIS_OPTIONS = [
    ("edge_canny", "エッジ検出（Canny）"),
    ("edge_sobel", "エッジ検出（Sobel）"),
    ("hough_lines", "Hough 線検出（確率的）"),
    ("hough_circles", "Hough 円検出"),
]
ANALYSIS_LABELS = dict(ANALYSIS_OPTIONS)
THRESHOLD_MODE_OPTIONS = [
    ("fixed", "固定閾値"),
    ("otsu", "Otsu"),
    ("adaptive_gaussian", "適応的（Gaussian）"),
]
THRESHOLD_MODE_LABELS = dict(THRESHOLD_MODE_OPTIONS)
ANALYSIS_PARAM_USAGE = {
    "edge_canny": {"canny_low", "canny_high"},
    "edge_sobel": {"sobel_ksize"},
    "hough_lines": {
        "line_canny_low",
        "line_canny_high",
        "hough_threshold",
        "hough_min_line_length",
        "hough_max_line_gap",
    },
    "hough_circles": {
        "circle_median_blur_kernel",
        "circle_dp",
        "circle_min_dist",
        "circle_param1",
        "circle_param2",
        "circle_min_radius",
        "circle_max_radius",
    },
}
THRESHOLD_MODE_PARAM_USAGE = {
    "fixed": {"threshold_value"},
    "otsu": set(),
    "adaptive_gaussian": {"adaptive_block_size", "adaptive_c"},
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
        "threshold_mode": "fixed",
        "threshold_value": 128,
        "adaptive_block_size": 11,
        "adaptive_c": 2,
    }


def default_pipeline_stages() -> list[dict[str, int | float | str | bool]]:
    return [default_pipeline_stage_values(i) for i in range(1, PIPELINE_STAGE_COUNT + 1)]


def default_analysis_settings() -> dict[str, int | float | str]:
    return {
        "analysis_type": "edge_canny",
        "canny_low": 80,
        "canny_high": 160,
        "sobel_ksize": 3,
        "line_canny_low": 80,
        "line_canny_high": 160,
        "hough_threshold": 50,
        "hough_min_line_length": 40,
        "hough_max_line_gap": 10,
        "circle_median_blur_kernel": 5,
        "circle_dp": 1.2,
        "circle_min_dist": 20,
        "circle_param1": 100,
        "circle_param2": 30,
        "circle_min_radius": 0,
        "circle_max_radius": 0,
    }


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


def normalize_threshold_mode(value: str) -> str:
    if value in THRESHOLD_MODE_LABELS:
        return value
    return "fixed"


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


def summarize_image(image: np.ndarray) -> dict[str, str]:
    return {
        "size": f"{image.shape[1]} x {image.shape[0]}",
        "channels": str(image.shape[2]) if image.ndim == 3 else "1",
        "dtype": str(image.dtype),
    }


def compute_ratio_metrics(binary_like: np.ndarray) -> dict[str, float | int]:
    nonzero_pixels = int(np.count_nonzero(binary_like))
    total_pixels = int(binary_like.size)
    ratio = (nonzero_pixels / total_pixels) if total_pixels else 0.0
    return {
        "nonzero_pixels": nonzero_pixels,
        "total_pixels": total_pixels,
        "ratio": ratio,
    }


def format_ratio(value: float) -> str:
    return f"{value:.4f} ({value * 100:.2f}%)"


def apply_threshold_mode(
    gray: np.ndarray,
    *,
    mode: str,
    threshold_value: int,
    adaptive_block_size: int,
    adaptive_c: int,
) -> tuple[np.ndarray, dict[str, str]]:
    params: dict[str, str] = {"threshold_mode": THRESHOLD_MODE_LABELS[mode]}

    if mode == "fixed":
        used_threshold, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        params["threshold_value"] = str(threshold_value)
        params["used_threshold"] = f"{used_threshold:.2f}"
        return binary, params

    if mode == "otsu":
        used_threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        params["used_threshold"] = f"{used_threshold:.2f}"
        return binary, params

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        adaptive_block_size,
        adaptive_c,
    )
    params["adaptive_block_size"] = str(adaptive_block_size)
    params["adaptive_c"] = str(adaptive_c)
    return binary, params


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
        stage["saturation_percent"] = parse_int(
            form,
            f"{prefix}_saturation_percent",
            int(stage["saturation_percent"]),
            min_value=0,
            max_value=300,
        )
        stage["blur_kernel"] = make_odd(
            parse_int(form, f"{prefix}_blur_kernel", int(stage["blur_kernel"]), min_value=1, max_value=99),
            minimum=1,
        )
        stage["threshold_mode"] = normalize_threshold_mode(
            str(form.get(f"{prefix}_threshold_mode", stage["threshold_mode"]))
        )
        stage["threshold_value"] = parse_int(
            form,
            f"{prefix}_threshold_value",
            int(stage["threshold_value"]),
            min_value=0,
            max_value=255,
        )
        stage["adaptive_block_size"] = make_odd(
            parse_int(
                form,
                f"{prefix}_adaptive_block_size",
                int(stage["adaptive_block_size"]),
                min_value=3,
                max_value=99,
            ),
            minimum=3,
        )
        stage["adaptive_c"] = parse_int(
            form,
            f"{prefix}_adaptive_c",
            int(stage["adaptive_c"]),
            min_value=-50,
            max_value=50,
        )
    return stages


def parse_analysis_settings(form) -> dict[str, int | float | str]:
    settings = default_analysis_settings()

    analysis_type = form.get("analysis_type", str(settings["analysis_type"]))
    if analysis_type not in ANALYSIS_LABELS:
        analysis_type = "edge_canny"
    settings["analysis_type"] = analysis_type

    settings["canny_low"] = parse_int(form, "canny_low", int(settings["canny_low"]), min_value=0, max_value=500)
    settings["canny_high"] = parse_int(form, "canny_high", int(settings["canny_high"]), min_value=1, max_value=500)
    if int(settings["canny_high"]) <= int(settings["canny_low"]):
        settings["canny_high"] = int(settings["canny_low"]) + 1

    settings["sobel_ksize"] = make_odd(
        parse_int(form, "sobel_ksize", int(settings["sobel_ksize"]), min_value=1, max_value=31),
        minimum=1,
    )

    settings["line_canny_low"] = parse_int(form, "line_canny_low", int(settings["line_canny_low"]), min_value=0, max_value=500)
    settings["line_canny_high"] = parse_int(form, "line_canny_high", int(settings["line_canny_high"]), min_value=1, max_value=500)
    if int(settings["line_canny_high"]) <= int(settings["line_canny_low"]):
        settings["line_canny_high"] = int(settings["line_canny_low"]) + 1
    settings["hough_threshold"] = parse_int(form, "hough_threshold", int(settings["hough_threshold"]), min_value=1, max_value=500)
    settings["hough_min_line_length"] = parse_int(
        form,
        "hough_min_line_length",
        int(settings["hough_min_line_length"]),
        min_value=1,
        max_value=2000,
    )
    settings["hough_max_line_gap"] = parse_int(
        form,
        "hough_max_line_gap",
        int(settings["hough_max_line_gap"]),
        min_value=0,
        max_value=500,
    )

    settings["circle_median_blur_kernel"] = make_odd(
        parse_int(
            form,
            "circle_median_blur_kernel",
            int(settings["circle_median_blur_kernel"]),
            min_value=1,
            max_value=99,
        ),
        minimum=1,
    )
    settings["circle_dp"] = parse_float(form, "circle_dp", float(settings["circle_dp"]), min_value=1.0, max_value=5.0)
    settings["circle_min_dist"] = parse_int(
        form,
        "circle_min_dist",
        int(settings["circle_min_dist"]),
        min_value=1,
        max_value=2000,
    )
    settings["circle_param1"] = parse_int(form, "circle_param1", int(settings["circle_param1"]), min_value=1, max_value=500)
    settings["circle_param2"] = parse_int(form, "circle_param2", int(settings["circle_param2"]), min_value=1, max_value=500)
    settings["circle_min_radius"] = parse_int(
        form,
        "circle_min_radius",
        int(settings["circle_min_radius"]),
        min_value=0,
        max_value=5000,
    )
    settings["circle_max_radius"] = parse_int(
        form,
        "circle_max_radius",
        int(settings["circle_max_radius"]),
        min_value=0,
        max_value=5000,
    )
    if int(settings["circle_max_radius"]) and int(settings["circle_max_radius"]) < int(settings["circle_min_radius"]):
        settings["circle_max_radius"] = int(settings["circle_min_radius"])

    return settings


def gamma_correct(image: np.ndarray, gamma_value: float) -> np.ndarray:
    table = np.array(
        [np.clip(((i / 255.0) ** gamma_value) * 255.0, 0, 255) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def apply_pipeline_stage(
    image: np.ndarray, stage: dict[str, int | float | str | bool]
) -> tuple[np.ndarray, dict[str, str]]:
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
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
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
    if operation == "threshold":
        gray = to_gray(image)
        mode = normalize_threshold_mode(str(stage["threshold_mode"]))
        threshold_value = int(stage["threshold_value"])
        adaptive_block_size = make_odd(int(stage["adaptive_block_size"]), minimum=3)
        adaptive_c = int(stage["adaptive_c"])
        binary, threshold_info = apply_threshold_mode(
            gray,
            mode=mode,
            threshold_value=threshold_value,
            adaptive_block_size=adaptive_block_size,
            adaptive_c=adaptive_c,
        )
        info.update(threshold_info)
        return binary, info
    if operation == "grayscale":
        return to_gray(image), info
    if operation == "invert":
        return 255 - image, info
    raise ValueError("未対応のパイプライン処理です。")


def run_pipeline(
    original: np.ndarray, stages: list[dict[str, int | float | str | bool]]
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    current = original.copy()
    executed_steps: list[dict[str, object]] = []
    stage_previews: list[dict[str, object]] = []

    for stage in stages:
        if not bool(stage["enabled"]):
            continue
        current, info = apply_pipeline_stage(current, stage)
        operation = str(stage["operation"])
        executed_steps.append(
            {
                "index": int(stage["index"]),
                "label": PIPELINE_OPERATION_LABELS[operation],
                "operation": operation,
                "info": info,
            }
        )
        stage_previews.append(
            {
                "index": int(stage["index"]),
                "label": PIPELINE_OPERATION_LABELS[operation],
                "src": encode_image_data_url(current),
                "image_info": summarize_image(current),
            }
        )

    return current, executed_steps, stage_previews


def analyze_edge_canny(image: np.ndarray, settings: dict[str, int | float | str]) -> dict[str, object]:
    gray = to_gray(image)
    low = int(settings["canny_low"])
    high = int(settings["canny_high"])
    edges = cv2.Canny(gray, low, high)
    metrics = compute_ratio_metrics(edges)
    status = "no_detection" if metrics["nonzero_pixels"] == 0 else "ok"
    return {
        "type": "edge_canny",
        "label": ANALYSIS_LABELS["edge_canny"],
        "status": status,
        "main_output": edges,
        "main_output_label": "エッジ画像",
        "debug_outputs": [],
        "summary": {
            "nonzero_pixels": str(metrics["nonzero_pixels"]),
            "edge_pixel_ratio": format_ratio(float(metrics["ratio"])),
        },
        "params": {
            "canny_low": str(low),
            "canny_high": str(high),
        },
    }


def analyze_edge_sobel(image: np.ndarray, settings: dict[str, int | float | str]) -> dict[str, object]:
    gray = to_gray(image)
    ksize = make_odd(int(settings["sobel_ksize"]), minimum=1)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(grad_x, grad_y)
    sobel_edges = cv2.convertScaleAbs(magnitude)

    metrics = compute_ratio_metrics(sobel_edges)
    status = "no_detection" if metrics["nonzero_pixels"] == 0 else "ok"

    return {
        "type": "edge_sobel",
        "label": ANALYSIS_LABELS["edge_sobel"],
        "status": status,
        "main_output": sobel_edges,
        "main_output_label": "Sobel エッジ画像",
        "debug_outputs": [
            {"label": "Sobel X", "image": cv2.convertScaleAbs(np.abs(grad_x))},
            {"label": "Sobel Y", "image": cv2.convertScaleAbs(np.abs(grad_y))},
        ],
        "summary": {
            "nonzero_pixels": str(metrics["nonzero_pixels"]),
            "edge_pixel_ratio": format_ratio(float(metrics["ratio"])),
        },
        "params": {
            "sobel_ksize": str(ksize),
        },
    }


def analyze_threshold(image: np.ndarray, settings: dict[str, int | float | str]) -> dict[str, object]:
    gray = to_gray(image)
    mode = str(settings["threshold_mode"])
    binary, params = apply_threshold_mode(
        gray,
        mode=mode,
        threshold_value=int(settings["threshold_value"]),
        adaptive_block_size=int(settings["adaptive_block_size"]),
        adaptive_c=int(settings["adaptive_c"]),
    )

    white_pixels = int(np.count_nonzero(binary == 255))
    total_pixels = int(binary.size)
    white_ratio = (white_pixels / total_pixels) if total_pixels else 0.0

    return {
        "type": "threshold",
        "label": ANALYSIS_LABELS["threshold"],
        "status": "ok",
        "main_output": binary,
        "main_output_label": "二値化結果",
        "debug_outputs": [],
        "summary": {
            "white_pixels": str(white_pixels),
            "white_pixel_ratio": format_ratio(white_ratio),
        },
        "params": params,
    }


def analyze_hough_lines(image: np.ndarray, settings: dict[str, int | float | str]) -> dict[str, object]:
    gray = to_gray(image)
    low = int(settings["line_canny_low"])
    high = int(settings["line_canny_high"])
    hough_threshold = int(settings["hough_threshold"])
    min_length = int(settings["hough_min_line_length"])
    max_gap = int(settings["hough_max_line_gap"])

    edges = cv2.Canny(gray, low, high)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_length,
        maxLineGap=max_gap,
    )

    overlay = to_bgr(image)
    line_count = 0
    if lines is not None:
        line_count = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1), (x2, y2), (20, 220, 20), 2)

    edge_metrics = compute_ratio_metrics(edges)
    status = "no_detection" if line_count == 0 else "ok"

    return {
        "type": "hough_lines",
        "label": ANALYSIS_LABELS["hough_lines"],
        "status": status,
        "main_output": overlay,
        "main_output_label": "線オーバーレイ",
        "debug_outputs": [
            {
                "label": "Canny エッジ画像",
                "image": edges,
            }
        ],
        "summary": {
            "detected_lines": str(line_count),
            "edge_pixel_ratio": format_ratio(float(edge_metrics["ratio"])),
        },
        "params": {
            "line_canny_low": str(low),
            "line_canny_high": str(high),
            "hough_threshold": str(hough_threshold),
            "hough_min_line_length": str(min_length),
            "hough_max_line_gap": str(max_gap),
        },
    }


def analyze_hough_circles(image: np.ndarray, settings: dict[str, int | float | str]) -> dict[str, object]:
    gray = to_gray(image)
    blur_kernel = int(settings["circle_median_blur_kernel"])
    blurred = cv2.medianBlur(gray, blur_kernel)

    dp = float(settings["circle_dp"])
    min_dist = int(settings["circle_min_dist"])
    param1 = int(settings["circle_param1"])
    param2 = int(settings["circle_param2"])
    min_radius = int(settings["circle_min_radius"])
    max_radius = int(settings["circle_max_radius"])

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    overlay = to_bgr(image)
    circle_count = 0
    if circles is not None:
        circles_rounded = np.round(circles[0]).astype(int)
        circle_count = len(circles_rounded)
        for x, y, radius in circles_rounded:
            cv2.circle(overlay, (x, y), radius, (0, 200, 255), 2)
            cv2.circle(overlay, (x, y), 2, (255, 60, 60), 2)

    status = "no_detection" if circle_count == 0 else "ok"

    return {
        "type": "hough_circles",
        "label": ANALYSIS_LABELS["hough_circles"],
        "status": status,
        "main_output": overlay,
        "main_output_label": "円オーバーレイ",
        "debug_outputs": [
            {
                "label": "平滑化後グレー画像",
                "image": blurred,
            }
        ],
        "summary": {
            "detected_circles": str(circle_count),
        },
        "params": {
            "circle_median_blur_kernel": str(blur_kernel),
            "circle_dp": f"{dp:.2f}",
            "circle_min_dist": str(min_dist),
            "circle_param1": str(param1),
            "circle_param2": str(param2),
            "circle_min_radius": str(min_radius),
            "circle_max_radius": str(max_radius),
        },
    }


def serialize_analysis_result(raw: dict[str, object], *, layout_class: str) -> dict[str, object]:
    debug_outputs = []
    for item in raw.get("debug_outputs", []):
        if not isinstance(item, dict):
            continue
        image = item.get("image")
        if image is None:
            continue
        debug_outputs.append(
            {
                "label": str(item.get("label", "補助画像")),
                "src": encode_image_data_url(image),
                "image_info": summarize_image(image),
            }
        )

    main_output = raw.get("main_output")
    if main_output is None:
        raise ValueError("本解析結果画像の生成に失敗しました。")

    status = str(raw.get("status", "ok"))
    status_message = None
    if status == "no_detection":
        status_message = "検出対象が見つかりませんでした。前処理または解析パラメータを調整して再実行してください。"

    return {
        "type": str(raw.get("type", "unknown")),
        "label": str(raw.get("label", "本解析")),
        "status": status,
        "status_message": status_message,
        "main_output_src": encode_image_data_url(main_output),
        "main_output_label": str(raw.get("main_output_label", "本解析結果")),
        "main_output_info": summarize_image(main_output),
        "debug_outputs": debug_outputs,
        "summary": {str(k): str(v) for k, v in dict(raw.get("summary", {})).items()},
        "params": {str(k): str(v) for k, v in dict(raw.get("params", {})).items()},
        "layout_class": layout_class,
    }


def normalize_submit_action(value: str) -> str:
    if value in {"pipeline_only", "pipeline_and_analysis"}:
        return value
    return "pipeline_and_analysis"


def run_analysis(image: np.ndarray, settings: dict[str, int | float | str]) -> dict[str, object]:
    analysis_type = str(settings["analysis_type"])
    if analysis_type == "edge_canny":
        return analyze_edge_canny(image, settings)
    if analysis_type == "edge_sobel":
        return analyze_edge_sobel(image, settings)
    if analysis_type == "hough_lines":
        return analyze_hough_lines(image, settings)
    if analysis_type == "hough_circles":
        return analyze_hough_circles(image, settings)
    raise ValueError("未対応の本解析タイプです。")


def build_template_context(**kwargs):
    context = {
        "pipeline_operation_options": PIPELINE_OPERATION_OPTIONS,
        "pipeline_stages": default_pipeline_stages(),
        "pipeline_param_usage": {k: sorted(v) for k, v in PIPELINE_PARAM_USAGE.items()},
        "analysis_options": ANALYSIS_OPTIONS,
        "analysis_settings": default_analysis_settings(),
        "analysis_param_usage": {k: sorted(v) for k, v in ANALYSIS_PARAM_USAGE.items()},
        "threshold_mode_options": THRESHOLD_MODE_OPTIONS,
        "threshold_mode_param_usage": {k: sorted(v) for k, v in THRESHOLD_MODE_PARAM_USAGE.items()},
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
    analysis_settings = parse_analysis_settings(request.form)
    submit_action = normalize_submit_action(str(request.form.get("submit_action", "pipeline_and_analysis")))

    file = request.files.get("image")
    has_new_upload = file is not None and file.filename != ""

    if not has_new_upload and LAST_UPLOADED_IMAGE_BYTES is None:
        return (
            render_template(
                "index.html",
                **build_template_context(
                    pipeline_stages=pipeline_stages,
                    analysis_settings=analysis_settings,
                    error="画像ファイルを選択してください。",
                ),
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

        preprocessed_image, executed_steps, stage_previews = run_pipeline(original, pipeline_stages)
        is_landscape = original.shape[1] >= original.shape[0]
        layout_class = "stack-vertical" if is_landscape else "stack-horizontal"
        analysis: dict[str, object] | None = None
        if submit_action == "pipeline_and_analysis":
            analysis_raw = run_analysis(preprocessed_image, analysis_settings)
            analysis = serialize_analysis_result(analysis_raw, layout_class=layout_class)
            analysis_input_gray = to_gray(preprocessed_image)
            analysis["input_gray_src"] = encode_image_data_url(analysis_input_gray)
            analysis["input_gray_label"] = "解析前グレースケール画像"
            analysis["input_gray_info"] = summarize_image(analysis_input_gray)

        result = {
            "filename": filename,
            "used_cached_image": not has_new_upload,
            "submit_action": submit_action,
            "original_src": encode_image_data_url(original),
            "final_src": encode_image_data_url(preprocessed_image),
            "executed_steps": executed_steps,
            "stage_previews": stage_previews,
            "original_info": summarize_image(original),
            "final_info": summarize_image(preprocessed_image),
            "preview_layout_class": layout_class,
            "stage_layout_class": layout_class,
            "analysis": analysis,
        }
    except ValueError as exc:
        return (
            render_template(
                "index.html",
                **build_template_context(
                    pipeline_stages=pipeline_stages,
                    analysis_settings=analysis_settings,
                    error=str(exc),
                ),
            ),
            400,
        )

    return render_template(
        "index.html",
        **build_template_context(
            pipeline_stages=pipeline_stages,
            analysis_settings=analysis_settings,
            result=result,
        ),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
