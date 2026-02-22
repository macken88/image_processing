## Image Processing GUI (Flask + uv)

ローカル環境で画像処理GUIアプリを開発するための最小構成です。

### セットアップ

```bash
uv sync
```

### 起動

```bash
uv run python main.py
```

ブラウザで `http://127.0.0.1:5000` を開いてください。

### 現在の機能

- Flask ベースのローカルWeb GUI
- 画像ファイルアップロード
- Pillow で画像サイズ・モード・フォーマットを確認
