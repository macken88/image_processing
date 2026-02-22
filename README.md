## Image Processing GUI (Flask + uv + OpenCV)

ローカル環境で画像処理GUIアプリを開発するための最小構成です。
現在は OpenCV ベースの前処理検証たたき台になっています。

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
- OpenCV による基礎処理の適用と比較表示
- 実装済み処理（たたき台）
  - グレースケール
  - ガウシアンぼかし
  - 固定閾値二値化 / 適応的二値化
  - Canny エッジ検出
  - Hough 線検出（確率的）
