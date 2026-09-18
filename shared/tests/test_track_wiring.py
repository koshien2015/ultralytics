"""track.py の設定と配線のテスト

track.py は上から下へ流れる台本なので import できない。代わりに構文木を読んで、
「フラグを立てたのに何も起きない」たぐいの取りこぼしを防ぐ。

実際の推論はここでは動かさない（GPUと実素材が要るため）。
"""

import ast
from pathlib import Path

TRACK_PATH = Path(__file__).resolve().parents[1] / "track.py"


def track_tree() -> ast.Module:
    return ast.parse(TRACK_PATH.read_text(encoding="utf-8"))


def constants() -> dict:
    """モジュール直下の定数（大文字の代入）を集める。"""
    values = {}
    for node in track_tree().body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return values


def called_attributes(name: str) -> set[str]:
    """`name.xxx(...)` の形で呼ばれているメソッド名。"""
    calls = set()
    for node in ast.walk(track_tree()):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == name
        ):
            calls.add(node.func.attr)
    return calls


class TestPoseExport:
    def test_export_is_on_by_default(self):
        """ENABLE_POSE を立てるだけで pose.json が出ること。

        書き出しに別のフラグが要ると、骨格は描かれているのにファイルが無い、
        という取りこぼしが起きる。
        """
        assert constants()["POSE_EXPORT"] is True

    def test_pose_estimation_stays_off_by_default(self):
        """姿勢推定そのものは重いので既定は False のまま。"""
        assert constants()["ENABLE_POSE"] is False

    def test_recorder_is_wired_into_the_loop_and_the_end(self):
        """溜める処理と書き出す処理の両方が呼ばれていること。"""
        assert {"record", "save"} <= called_attributes("recorder")

    def test_recorder_is_guarded_by_identity_not_truthiness(self):
        """`if recorder:` だと、記録0件のあいだ recorder が falsy で素通りする。"""
        source = TRACK_PATH.read_text(encoding="utf-8")

        assert "if recorder:" not in source
        assert "elif recorder:" not in source
        assert "recorder is not None" in source

    def test_flush_interval_fits_a_short_clip(self):
        """切り出した数十フレームのクリップでも、途中経過が1回は書かれること。"""
        assert 0 < constants()["POSE_EXPORT_FLUSH_FRAMES"] <= 90

    def test_pose_export_module_is_imported(self):
        imported = {
            alias.name
            for node in ast.walk(track_tree())
            if isinstance(node, ast.Import)
            for alias in node.names
        }

        assert "pose_export" in imported
