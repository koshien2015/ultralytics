"""pose_export のユニットテスト

推論は要らない。溜めた結果の JSON の形、人物の選び方の記録、
区間の扱いだけを見る（test_pose.py と同じく ultralytics / torch は使わない）。
"""

import json

import numpy as np
import pytest

from pose import KEYPOINT_NAMES, PersonPose
from pose_export import SCHEMA_VERSION, PoseRecorder, gate_windows


def make_person(track_id=1, conf=0.9, offset=0.0):
    xs = np.linspace(100 + offset, 300 + offset, len(KEYPOINT_NAMES))
    ys = np.linspace(100, 400, len(KEYPOINT_NAMES))
    return PersonPose(
        track_id=track_id,
        keypoints=np.stack([xs, ys], axis=1),
        scores=np.full(len(KEYPOINT_NAMES), conf),
    )


def make_recorder(**kwargs):
    defaults = dict(
        keypoint_names=KEYPOINT_NAMES, fps=60.0, video_path="input/a.mov", min_score=0.5
    )
    return PoseRecorder(**{**defaults, **kwargs})


def result_with(person, role="pitcher"):
    return {"persons": [person], "roles": {person.track_id: role}}


class TestRecord:
    def test_records_one_frame_per_call(self):
        recorder = make_recorder()
        person = make_person()

        for frame in range(100, 105):
            recorder.record(frame, result_with(person))

        assert len(recorder) == 5

    def test_selection_mode_is_reported(self):
        recorder = make_recorder()

        mode = recorder.record(100, result_with(make_person()))

        assert mode == "role_bbox"
        assert recorder.selection_modes == ("role_bbox",)

    def test_missing_person_is_recorded_as_null(self):
        """人物が取れなかったフレームも、番号を飛ばさず null で残す。"""
        recorder = make_recorder()

        recorder.record(100, {"persons": [], "roles": {}})

        payload = recorder._payload()
        assert payload["frames"][0]["keypoints"]["nose"] == [None, None, 0.0]

    def test_all_keypoints_are_written_with_their_scores(self):
        """低信頼度でも捨てない。落とすかどうかは解析側が決める。"""
        recorder = make_recorder()

        recorder.record(100, result_with(make_person(conf=0.05)))

        keypoints = recorder._payload()["frames"][0]["keypoints"]
        assert len(keypoints) == len(KEYPOINT_NAMES)
        assert keypoints["nose"][2] == pytest.approx(0.05)


class TestFrameRanges:
    def test_continuous_frames_are_one_range(self):
        recorder = make_recorder()
        for frame in range(100, 105):
            recorder.record(frame, result_with(make_person()))

        assert recorder.frame_ranges() == [(100, 104)]

    def test_gaps_split_the_ranges(self):
        """推論を掛けなかったフレームは記録しないので、番号が飛ぶ。"""
        recorder = make_recorder()
        for frame in [100, 101, 102, 200, 201]:
            recorder.record(frame, result_with(make_person()))

        assert recorder.frame_ranges() == [(100, 102), (200, 201)]


class TestSave:
    def test_writes_the_schema_the_analyzer_reads(self, tmp_path):
        recorder = make_recorder(pitch_id="throw_01")
        for frame in range(100, 103):
            recorder.record(frame, result_with(make_person()))

        path = recorder.save(tmp_path / "a_pose.json")
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["meta"]["pitch_id"] == "throw_01"
        assert payload["meta"]["fps"] == 60.0
        assert len(payload["frames"]) == 3
        assert payload["frames"][0]["frame_index"] == 100
        assert payload["frames"][0]["timestamp_sec"] == pytest.approx(100 / 60.0)

    def test_pitch_id_defaults_to_the_video_name(self, tmp_path):
        recorder = make_recorder(video_path="input/throw_07.mov")
        recorder.record(100, result_with(make_person()))

        payload = json.loads(recorder.save(tmp_path / "x.json").read_text(encoding="utf-8"))

        assert payload["meta"]["pitch_id"] == "throw_07"

    def test_fallback_selection_is_warned_about(self, tmp_path):
        recorder = make_recorder()
        recorder.record(100, {"persons": [make_person()], "roles": {}})

        notes = json.loads(recorder.save(tmp_path / "x.json").read_text(encoding="utf-8"))
        joined = " ".join(notes["meta"]["notes"])

        assert "役割bbox" in joined, "誰の骨格か確定していないことが伝わらない"

    def test_split_ranges_are_warned_about(self, tmp_path):
        recorder = make_recorder()
        for frame in [100, 101, 300, 301]:
            recorder.record(frame, result_with(make_person()))

        notes = json.loads(recorder.save(tmp_path / "x.json").read_text(encoding="utf-8"))
        joined = " ".join(notes["meta"]["notes"])

        assert "100-101" in joined and "300-301" in joined
        assert "--start" in joined, "どう切り出せばよいか分からない"

    def test_saving_nothing_is_an_error(self, tmp_path):
        with pytest.raises(ValueError):
            make_recorder().save(tmp_path / "x.json")


class TestGateWindows:
    """推論を掛ける区間の決め方。None は「区間指定なし＝全フレーム」。"""

    def test_windows_are_used_when_they_exist(self):
        windows = ((100, 200), (400, 500))

        assert gate_windows(windows, export_enabled=True) is windows

    def test_empty_windows_fall_back_to_all_frames_when_exporting(self):
        """切り出し済みクリップでは窓が立たない。黙って0フレームにしない。"""
        assert gate_windows((), export_enabled=True) is None

    def test_empty_windows_stay_empty_when_not_exporting(self):
        """描画だけのときは従来どおり。全フレーム推論して遅くしない。"""
        assert gate_windows((), export_enabled=False) == ()
