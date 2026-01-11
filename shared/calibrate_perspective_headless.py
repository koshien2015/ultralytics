"""
射影変換キャリブレーション用ツール（ヘッドレス環境対応）

Docker/Colab環境でGUIが使えない場合に使用

使い方:
1. python calibrate_perspective_headless.py <video_file> [--frame FRAME_NUMBER]
2. フレーム画像が保存されます
3. ターミナルで座標を入力
4. 確認用画像とキャリブレーションJSONが生成されます
"""

import cv2
import sys
import os
import argparse
from perspective_transform import PerspectiveTransformer, get_default_field_points, BASEBALL_FIELD


def save_frame(video_path, frame_number, output_path):
    """動画から指定フレームを保存"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ エラー: 動画を開けませんでした: {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ エラー: フレーム {frame_number} を読み込めませんでした")
        return None

    cv2.imwrite(output_path, frame)
    print(f"✅ フレームを保存しました: {output_path}")
    return frame


def draw_calibration_points(image, points, labels, output_path):
    """キャリブレーション点を描画して保存"""
    display_image = image.copy()

    for i, (x, y) in enumerate(points):
        # 点を描画
        cv2.circle(display_image, (x, y), 8, (0, 255, 0), -1)
        cv2.circle(display_image, (x, y), 8, (255, 255, 255), 2)

        # ラベルを描画
        label_text = f"{i+1}: {labels[i]}"
        cv2.putText(display_image, label_text, (x + 10, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 点を線で結ぶ
    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(display_image, points[i], points[i+1], (255, 255, 0), 2)

    cv2.imwrite(output_path, display_image)
    print(f"✅ 確認用画像を保存しました: {output_path}")


def input_points_interactive():
    """ターミナルで対話的に座標を入力"""
    print("\n" + "="*60)
    print("キャリブレーション座標入力")
    print("="*60)
    print("\n基準点を4点以上入力してください")
    print("推奨: ホームベース、マウンド、1塁、3塁など")
    print("\n利用可能な基準点:")

    default_points = get_default_field_points()
    point_names = list(default_points.keys())

    for i, name in enumerate(point_names):
        coord = default_points[name]
        print(f"  {i+1}. {name:15s} (X={coord[0]:6.2f}m, Z={coord[1]:6.2f}m)")
    print(f"  {len(point_names)+1}. custom (手動入力)")

    image_points = []
    point_labels = []
    real_points = []

    while True:
        print(f"\n--- 点 {len(image_points)+1} ---")

        # 画像座標を入力
        try:
            pixel_x = int(input("画像上のX座標（ピクセル）: "))
            pixel_y = int(input("画像上のY座標（ピクセル）: "))
        except ValueError:
            print("⚠️  整数を入力してください")
            continue

        # 基準点の種類を選択
        print("\nこの点の種類を選択してください:")
        for i, name in enumerate(point_names):
            coord = default_points[name]
            print(f"  {i+1}. {name}")
        print(f"  {len(point_names)+1}. custom")

        while True:
            try:
                choice = int(input(f"選択 (1-{len(point_names)+1}): "))
                if 1 <= choice <= len(point_names)+1:
                    break
                print(f"1〜{len(point_names)+1}の数字を入力してください")
            except ValueError:
                print("数字を入力してください")

        # 実座標を取得
        if choice <= len(point_names):
            # 既定の基準点
            selected_name = point_names[choice - 1]
            real_coord = default_points[selected_name]
        else:
            # カスタム座標
            selected_name = "custom"
            try:
                real_x = float(input("実座標 X（メートル、ホームベース中心=0）: "))
                real_z = float(input("実座標 Z（メートル、ホームベース=0）: "))
                real_coord = (real_x, real_z)
            except ValueError:
                print("⚠️  数値を入力してください")
                continue

        # 追加
        image_points.append((pixel_x, pixel_y))
        point_labels.append(selected_name)
        real_points.append(real_coord)

        print(f"✓ 点 {len(image_points)} を追加: ({pixel_x}, {pixel_y}) → {selected_name} {real_coord}")

        # 継続確認
        if len(image_points) >= 4:
            cont = input(f"\n点を追加しますか？ (y/n, 現在{len(image_points)}点): ").lower()
            if cont != 'y':
                break
        else:
            print(f"（最低4点必要です。現在{len(image_points)}点）")

    return image_points, point_labels, real_points


def input_points_from_json_template(template_path):
    """JSONテンプレートから座標を読み込み"""
    import json

    print(f"\nJSONテンプレートを読み込んでいます: {template_path}")

    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_points = [tuple(pt) for pt in data['image_points']]
        real_points = [tuple(pt) for pt in data['real_points']]
        point_labels = data.get('labels', ['point' for _ in image_points])

        print(f"✅ {len(image_points)}点を読み込みました")
        return image_points, point_labels, real_points
    except Exception as e:
        print(f"❌ エラー: JSONファイルの読み込みに失敗しました: {e}")
        return None, None, None


def create_json_template(output_path):
    """JSONテンプレートを作成"""
    template = {
        "image_points": [
            [640, 720],  # ホームベース（例）
            [640, 300],  # マウンド（例）
            [500, 150],  # 1塁（例）
            [780, 150]   # 3塁（例）
        ],
        "real_points": [
            [0.0, 0.0],      # ホームベース
            [0.0, 9.22],    # マウンド
            [27.43, 27.43],  # 1塁
            [-27.43, 27.43]  # 3塁
        ],
        "labels": [
            "home_plate",
            "pitcher_mound",
            "first_base",
            "third_base"
        ],
        "note": "image_points（画像座標）とreal_points（実座標）を編集してください"
    }

    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"✅ JSONテンプレートを作成しました: {output_path}")
    print(f"\nこのファイルを編集してから、--json オプションで読み込んでください:")
    print(f"  python {sys.argv[0]} {sys.argv[1]} --json {output_path}")


def main():
    parser = argparse.ArgumentParser(description='射影変換キャリブレーションツール（ヘッドレス環境対応）')
    parser.add_argument('video', help='動画ファイルパス')
    parser.add_argument('--frame', type=int, default=0, help='キャリブレーションに使用するフレーム番号')
    parser.add_argument('--json', help='JSONテンプレートから座標を読み込み')
    parser.add_argument('--create-template', action='store_true', help='JSONテンプレートを作成して終了')
    args = parser.parse_args()

    # 動画パスの確認
    if not os.path.exists(args.video):
        print(f"❌ エラー: 動画ファイルが見つかりません: {args.video}")
        sys.exit(1)

    # ファイル名とディレクトリを取得
    video_dir = os.path.dirname(args.video) if os.path.dirname(args.video) else "."
    base_name = os.path.splitext(os.path.basename(args.video))[0]

    # フレームを保存
    frame_image_path = os.path.join(video_dir, f"{base_name}_frame{args.frame}.jpg")
    frame = save_frame(args.video, args.frame, frame_image_path)

    if frame is None:
        sys.exit(1)

    height, width = frame.shape[:2]
    print(f"\n動画情報: {width}x{height}")
    print(f"保存されたフレーム画像を確認してください: {frame_image_path}")

    # JSONテンプレート作成モード
    if args.create_template:
        template_path = os.path.join(video_dir, f"{base_name}_calibration_template.json")
        create_json_template(template_path)
        return

    # 座標を入力
    if args.json:
        # JSONから読み込み
        image_points, point_labels, real_points = input_points_from_json_template(args.json)
        if image_points is None:
            sys.exit(1)
    else:
        # 対話的に入力
        print("\n保存されたフレーム画像を開いて、基準点の座標を確認してください。")
        image_points, point_labels, real_points = input_points_interactive()

    # 確認用画像を保存
    preview_path = os.path.join(video_dir, f"{base_name}_calibration_preview.jpg")
    draw_calibration_points(frame, image_points, point_labels, preview_path)

    # キャリブレーション実行
    transformer = PerspectiveTransformer()
    if transformer.calibrate(image_points, real_points):
        # 保存
        output_path = os.path.join(video_dir, f"{base_name}_calibration.json")
        if transformer.save_calibration(output_path):
            print(f"\n{'='*60}")
            print(f"✅ キャリブレーション完了！")
            print(f"{'='*60}")
            print(f"保存先: {output_path}")
            print(f"確認用画像: {preview_path}")
            print(f"登録点数: {len(image_points)}")
            print(f"\n次のステップ:")
            print(f"1. 確認用画像で位置を確認")
            print(f"2. track.pyのCALIBRATION_FILEを設定")
            print(f"3. python track.py {args.video}")
            print(f"{'='*60}\n")
    else:
        print("❌ キャリブレーションに失敗しました")
        sys.exit(1)


if __name__ == "__main__":
    main()
