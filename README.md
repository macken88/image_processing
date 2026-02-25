## Image Processing GUI (Flask + uv + OpenCV)

ローカル環境で画像処理GUIアプリを開発するための最小構成です。
現在は OpenCV ベースの前処理 + 本解析（Canny/Sobel/Hough）検証たたき台になっています。

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
- OpenCV による基本処理の前処理パイプライン実行（複数ステージ）
- 前処理結果に対して本解析を1種類選択して実行（手動反復で比較）
- 原画像 / 前処理結果 / 本解析結果 / 各ステージ後の途中結果を比較表示
- 実行された前処理ステップと、本解析の判定補助指標・実適用パラメータを表示
- 実装済みの基本処理（パイプライン用）
  - 明るさ調整
  - コントラスト調整
  - ガンマ変換
  - 色相シフト
  - 彩度調整
  - ガウシアンぼかし
  - 二値化（固定閾値 / Otsu / 適応的）
  - グレースケール
  - 反転
- 実装済みの本解析（v1）
  - エッジ検出（Canny）
  - エッジ検出（Sobel）
  - Hough 線検出（確率的）
  - Hough 円検出
- 画像を再選択しなくても、前回アップロード画像を再利用して前処理/本解析を再実行可能

### 設定できるパラメータ（前処理パイプライン）

- 明るさ調整（`brightness`）
  - `brightness`: 画素値に加算する値（負値で暗く、正値で明るく）
- コントラスト調整（`contrast`）
  - `contrast`: コントラスト倍率（`1.0` で変化なし）
- ガンマ変換（`gamma`）
  - `gamma`: ガンマ値（小さくすると明るめ、大きくすると暗め）
- 色相シフト（`hue_shift`）
  - `hue_deg`: 色相を回転させる角度（度）
- 彩度調整（`saturation`）
  - `saturation_percent`: 彩度倍率（`100` で変化なし、`0` で無彩色）
- ガウシアンぼかし（`gaussian_blur`）
  - `blur_kernel`: カーネルサイズ（奇数、内部で奇数に補正）
- 二値化（`threshold`）
  - `threshold_mode`: 二値化モード（`fixed` / `otsu` / `adaptive_gaussian`）
  - `threshold_value`: 固定閾値（`threshold_mode=fixed` で使用）
  - `adaptive_block_size`: 適応的二値化の近傍サイズ（奇数、`adaptive_gaussian` で使用）
  - `adaptive_c`: 適応的二値化の補正値（`adaptive_gaussian` で使用）
- グレースケール（`grayscale`）
  - パラメータなし
- 反転（`invert`）
  - パラメータなし

### 設定できるパラメータ（本解析）

- エッジ検出（Canny / `edge_canny`）
  - `canny_low`: Canny の低閾値
  - `canny_high`: Canny の高閾値（`low` より大きくなるよう補正）

- エッジ検出（Sobel / `edge_sobel`）
  - `sobel_ksize`: Sobel カーネルサイズ（奇数）

- Hough 線検出（`hough_lines`）
  - `line_canny_low`: 線検出前の Canny 低閾値
  - `line_canny_high`: 線検出前の Canny 高閾値（`low` より大きくなるよう補正）
  - `hough_threshold`: Hough 変換の投票閾値
  - `hough_min_line_length`: 検出する線分の最小長
  - `hough_max_line_gap`: 線分をつなぐ最大ギャップ

- Hough 円検出（`hough_circles`）
  - `circle_median_blur_kernel`: 円検出前の median blur カーネル（奇数）
  - `circle_dp`: Hough 円検出の解像度比
  - `circle_min_dist`: 検出円どうしの最小中心距離
  - `circle_param1`: 内部 Canny 高閾値相当
  - `circle_param2`: 円中心検出の閾値（小さいほど検出されやすい）
  - `circle_min_radius`: 検出する円の最小半径
  - `circle_max_radius`: 検出する円の最大半径（`0` は制限なし）
