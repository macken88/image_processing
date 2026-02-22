from pathlib import Path

from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/inspect")
def inspect_image():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", error="画像ファイルを選択してください。"), 400

    try:
        image = Image.open(file.stream)
        width, height = image.size
        mode = image.mode
        fmt = image.format or "Unknown"
    except UnidentifiedImageError:
        return render_template("index.html", error="画像として読み込めないファイルです。"), 400

    return render_template(
        "index.html",
        result={
            "filename": file.filename,
            "width": width,
            "height": height,
            "mode": mode,
            "format": fmt,
        },
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
