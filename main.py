import base64
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

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

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB


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


def parse_int(form, key: str, default: int, *, min_value: int, max_value: int) -> int:
    raw = form.get(key, "")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def make_odd(value: int, *, minimum: int) -> int:
    if value < minimum:
        value = minimum
    if value % 2 == 0:
        value += 1
    return value


def decode_uploaded_image(file_storage) -> np.ndarray:
    data = file_storage.read()
    if not data:
        raise ValueError("画像データが空です。")

    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("画像として読み込めないファイルです。")
    return image


def encode_image_data_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("画像の表示用エンコードに失敗しました。")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


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


def build_template_context(**kwargs):
    context = {
        "process_options": PROCESS_OPTIONS,
        "form_values": default_form_values(),
        "error": None,
        "result": None,
    }
    context.update(kwargs)
    return context


@app.get("/")
def index():
    return render_template("index.html", **build_template_context())


@app.post("/process")
def process_image():
    form_values = default_form_values()
    form_values["process_type"] = request.form.get("process_type", "grayscale")
    if form_values["process_type"] not in PROCESS_LABELS:
        form_values["process_type"] = "grayscale"

    form_values["blur_kernel"] = make_odd(
        parse_int(request.form, "blur_kernel", 5, min_value=1, max_value=99),
        minimum=1,
    )
    form_values["threshold_value"] = parse_int(
        request.form, "threshold_value", 128, min_value=0, max_value=255
    )
    form_values["adaptive_block_size"] = make_odd(
        parse_int(request.form, "adaptive_block_size", 11, min_value=3, max_value=99),
        minimum=3,
    )
    form_values["adaptive_c"] = parse_int(request.form, "adaptive_c", 2, min_value=-50, max_value=50)
    form_values["canny_low"] = parse_int(request.form, "canny_low", 80, min_value=0, max_value=500)
    form_values["canny_high"] = parse_int(request.form, "canny_high", 160, min_value=1, max_value=500)
    form_values["hough_threshold"] = parse_int(
        request.form, "hough_threshold", 50, min_value=1, max_value=500
    )
    form_values["hough_min_line_length"] = parse_int(
        request.form, "hough_min_line_length", 40, min_value=1, max_value=2000
    )
    form_values["hough_max_line_gap"] = parse_int(
        request.form, "hough_max_line_gap", 10, min_value=0, max_value=500
    )

    file = request.files.get("image")
    if file is None or file.filename == "":
        return (
            render_template(
                "index.html",
                **build_template_context(
                    form_values=form_values,
                    error="画像ファイルを選択してください。",
                ),
            ),
            400,
        )

    try:
        original = decode_uploaded_image(file)
        processed, process_info = apply_process(original, form_values)
        result = {
            "filename": file.filename,
            "original_src": encode_image_data_url(original),
            "processed_src": encode_image_data_url(processed),
            "process_label": PROCESS_LABELS[str(form_values["process_type"])],
            "image_info": {
                "original_size": f"{original.shape[1]} x {original.shape[0]}",
                "original_channels": str(original.shape[2]) if original.ndim == 3 else "1",
                "processed_size": f"{processed.shape[1]} x {processed.shape[0]}",
                "processed_channels": str(processed.shape[2]) if processed.ndim == 3 else "1",
                "dtype": str(processed.dtype),
            },
            "process_info": process_info,
        }
    except ValueError as exc:
        return (
            render_template(
                "index.html",
                **build_template_context(form_values=form_values, error=str(exc)),
            ),
            400,
        )

    return render_template(
        "index.html",
        **build_template_context(form_values=form_values, result=result),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
