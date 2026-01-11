"""
射影変換キャリブレーション用UIツール

使い方:
1. python calibrate_perspective.py <video_file> [--frame FRAME_NUMBER]
2. 画像上で基準点を4点以上クリック（ホームベース、マウンド、ベースなど）
3. 各点の実座標を入力
4. キャリブレーションデータをJSONファイルに保存

キー操作:
- クリック: 基準点を追加
- 'u': 最後の点を削除（Undo）
- 'r': 全ての点をリセット
- 's': キャリブレーションを保存
- 'q': 終了
"""

import cv2
import sys
import os
import argparse
from perspective_transform import PerspectiveTransformer, get_default_field_points, BASEBALL_FIELD


class CalibrationUI:
    """キャリブレーション用UIクラス"""

    def __init__(self, image, video_path):
        self.image = image.copy()
        self.display_image = image.copy()
        self.video_path = video_path
        self.points = []  # [(x, y), ...]
        self.point_labels = []  # ['home_plate', 'pitcher_mound', ...]
        self.window_name = 'Perspective Calibration - Click points on field'

        # デフォルトの基準点名と座標
        self.default_points = get_default_field_points()
        self.available_labels = list(self.default_points.keys()) + ['custom']

        print("\n" + "="*60)
        print("射影変換キャリブレーションツール")
        print("="*60)
        print("\n【使い方】")
        print("1. 画像上で基準点を4点以上クリックしてください")
        print("2. 推奨: ホームベース、マウンド、1塁、3塁など")
        print("3. 各点をクリック後、ターミナルで点の種類を選択")
        print("4. 4点以上選択したら 's' キーで保存")
        print("\n【キー操作】")
        print("  クリック: 基準点を追加")
        print("  u: 最後の点を削除（Undo）")
        print("  r: 全ての点をリセット")
        print("  s: キャリブレーションを保存（4点以上必要）")
        print("  q: 終了")
        print("="*60 + "\n")

    def mouse_callback(self, event, x, y, flags, param):
        """マウスクリックコールバック"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_point(x, y)

    def add_point(self, x, y):
        """基準点を追加"""
        self.points.append((x, y))

        # ユーザーに点の種類を選択させる
        print(f"\n点 {len(self.points)} を追加: ({x}, {y})")
        print("この点の種類を選択してください:")
        for i, label in enumerate(self.available_labels):
            if label in self.default_points:
                coord = self.default_points[label]
                print(f"  {i+1}. {label} (X={coord[0]:.2f}m, Z={coord[1]:.2f}m)")
            else:
                print(f"  {i+1}. {label} (手動入力)")

        while True:
            try:
                choice = input(f"選択 (1-{len(self.available_labels)}): ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(self.available_labels):
                    selected_label = self.available_labels[choice_idx]
                    self.point_labels.append(selected_label)
                    print(f"✓ '{selected_label}' として登録しました")
                    break
                else:
                    print(f"1〜{len(self.available_labels)}の数字を入力してください")
            except ValueError:
                print("数字を入力してください")

        self.update_display()

    def undo_last_point(self):
        """最後の点を削除"""
        if self.points:
            removed = self.points.pop()
            label = self.point_labels.pop()
            print(f"✓ 最後の点を削除しました: {removed} ({label})")
            self.update_display()
        else:
            print("削除する点がありません")

    def reset_points(self):
        """全ての点をリセット"""
        self.points = []
        self.point_labels = []
        print("✓ 全ての点をリセットしました")
        self.update_display()

    def update_display(self):
        """表示を更新"""
        self.display_image = self.image.copy()

        # 点を描画
        for i, (x, y) in enumerate(self.points):
            cv2.circle(self.display_image, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(self.display_image, f"{i+1}: {self.point_labels[i]}",
                       (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 4点以上の場合は線で結ぶ
        if len(self.points) >= 4:
            for i in range(len(self.points) - 1):
                cv2.line(self.display_image, self.points[i], self.points[i+1], (255, 255, 0), 2)

        # ステータス表示
        status = f"Points: {len(self.points)}/4+ (Press 's' to save, 'u' to undo, 'r' to reset)"
        cv2.putText(self.display_image, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(self.window_name, self.display_image)

    def save_calibration(self):
        """キャリブレーションを保存"""
        if len(self.points) < 4:
            print(f"❌ エラー: 最低4点必要です（現在{len(self.points)}点）")
            return False

        # フィールド座標を取得
        real_points = []
        for label in self.point_labels:
            if label in self.default_points:
                real_points.append(self.default_points[label])
            else:
                # カスタム座標を入力
                print(f"\n'{label}' の実座標を入力してください:")
                while True:
                    try:
                        x = float(input("  X座標 (m): "))
                        z = float(input("  Z座標 (m): "))
                        real_points.append((x, z))
                        break
                    except ValueError:
                        print("数値を入力してください")

        # PerspectiveTransformerでキャリブレーション
        transformer = PerspectiveTransformer()
        if transformer.calibrate(self.points, real_points):
            # 保存先を決定
            video_dir = os.path.dirname(self.video_path) if os.path.dirname(self.video_path) else "."
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            output_path = os.path.join(video_dir, f"{base_name}_calibration.json")

            if transformer.save_calibration(output_path):
                print(f"\n{'='*60}")
                print(f"✅ キャリブレーション完了！")
                print(f"{'='*60}")
                print(f"保存先: {output_path}")
                print(f"登録点数: {len(self.points)}")
                print(f"\n次のステップ:")
                print(f"  python track.py {self.video_path}")
                print(f"{'='*60}\n")
                return True

        return False

    def run(self):
        """UIを起動"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.update_display()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("終了します")
                break
            elif key == ord('u'):
                self.undo_last_point()
            elif key == ord('r'):
                self.reset_points()
            elif key == ord('s'):
                if self.save_calibration():
                    break

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='射影変換キャリブレーションツール')
    parser.add_argument('video', help='動画ファイルパス')
    parser.add_argument('--frame', type=int, default=0, help='キャリブレーションに使用するフレーム番号')
    args = parser.parse_args()

    # 動画を開く
    if not os.path.exists(args.video):
        print(f"❌ エラー: 動画ファイルが見つかりません: {args.video}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ エラー: 動画を開けませんでした: {args.video}")
        sys.exit(1)

    # 指定フレームを取得
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ エラー: フレーム {args.frame} を読み込めませんでした")
        sys.exit(1)

    print(f"✓ フレーム {args.frame} を読み込みました")

    # UIを起動
    ui = CalibrationUI(frame, args.video)
    ui.run()


if __name__ == "__main__":
    main()
