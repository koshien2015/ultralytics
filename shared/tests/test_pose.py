"""pose のユニットテスト

ultralytics / torch を必要としない純粋ロジック（役割マッピング・骨格の
役割割り当て・ID引き継ぎ・窓設定の導出・描画）だけを対象にする。
YOLO 推論そのものは実素材が要るのでここでは検証しない。
"""

import numpy as np
import pytest

from prefilter import PrefilterConfig
from pose import (
    KEYPOINT_NAMES,
    select_person,
    SKELETON_EDGES,
    PersonPose,
    RoleTracker,
    build_role_boxes,
    draw_skeleton,
    keypoint_bbox,
    overlap_coefficient,
    pose_prefilter_config,
    role_of_detection,
)


def make_person(track_id, x1, y1, x2, y2, conf=0.9):
    """bbox の内側に17点を散らした PersonPose を作る"""
    xs = np.linspace(x1 + 1, x2 - 1, len(KEYPOINT_NAMES))
    ys = np.linspace(y1 + 1, y2 - 1, len(KEYPOINT_NAMES))
    keypoints = np.stack([xs, ys], axis=1)
    scores = np.full(len(KEYPOINT_NAMES), conf, dtype=np.float64)
    return PersonPose(track_id=track_id, keypoints=keypoints, scores=scores)


class TestRoleOfDetection:
    def test_pitcher_classes_map_to_pitcher(self):
        assert role_of_detection(1) == "pitcher"   # pitcher_motion
        assert role_of_detection(5) == "pitcher"   # pitcher_release

    def test_batter_classes_map_to_batter(self):
        assert role_of_detection(2) == "batter"    # batter_stance
        assert role_of_detection(6) == "batter"    # batter_swing

    def test_all_catcher_states_map_to_catcher(self):
        for cls in (4, 7, 8, 9, 10):
            assert role_of_detection(cls) == "catcher"

    def test_umpire(self):
        assert role_of_detection(3) == "umpire"

    def test_cap_is_not_a_person(self):
        assert role_of_detection(0) is None

    def test_unknown_class(self):
        assert role_of_detection(99) is None


class TestOverlapCoefficient:
    def test_identical_boxes(self):
        box = (0, 0, 10, 10)
        assert overlap_coefficient(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert overlap_coefficient((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_touching_boxes_do_not_overlap(self):
        assert overlap_coefficient((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0

    def test_small_box_fully_inside_large_one_scores_one(self):
        # 包含係数は min(面積) で割るので、サイズ差があっても1.0になる
        assert overlap_coefficient((0, 0, 100, 100), (10, 10, 20, 20)) == pytest.approx(1.0)

    def test_partial_overlap(self):
        # 交差 5x10=50、小さい方の面積 10x10=100
        assert overlap_coefficient((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(0.5)

    def test_zero_area_box_is_safe(self):
        assert overlap_coefficient((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


class TestKeypointBbox:
    def test_bbox_covers_confident_keypoints(self):
        keypoints = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        scores = np.array([0.9, 0.9, 0.9])
        assert keypoint_bbox(keypoints, scores, min_score=0.5) == (10.0, 20.0, 50.0, 60.0)

    def test_low_confidence_keypoints_are_excluded(self):
        keypoints = np.array([[10.0, 20.0], [30.0, 40.0], [999.0, 999.0]])
        scores = np.array([0.9, 0.9, 0.1])
        assert keypoint_bbox(keypoints, scores, min_score=0.5) == (10.0, 20.0, 30.0, 40.0)

    def test_returns_none_when_nothing_is_confident(self):
        keypoints = np.array([[10.0, 20.0], [30.0, 40.0]])
        scores = np.array([0.1, 0.2])
        assert keypoint_bbox(keypoints, scores, min_score=0.5) is None


class FakeBox:
    def __init__(self, cls, xyxy):
        self.cls = [cls]
        self.xyxy = [list(xyxy)]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class TestBuildRoleBoxes:
    def test_collects_one_box_per_role(self):
        results = [FakeResult([
            FakeBox(1, (100, 100, 200, 300)),   # pitcher_motion
            FakeBox(2, (600, 150, 680, 350)),   # batter_stance
            FakeBox(0, (400, 200, 410, 210)),   # cap: 人物ではない
        ])]
        boxes = build_role_boxes(results)
        assert set(boxes) == {"pitcher", "batter"}
        assert boxes["pitcher"] == (100.0, 100.0, 200.0, 300.0)

    def test_first_detection_of_a_role_wins(self):
        results = [FakeResult([
            FakeBox(1, (100, 100, 200, 300)),
            FakeBox(5, (700, 700, 800, 900)),   # 同じ pitcher ロール
        ])]
        assert build_role_boxes(results)["pitcher"] == (100.0, 100.0, 200.0, 300.0)

    def test_empty_results(self):
        assert build_role_boxes([]) == {}


class TestRoleTracker:
    def test_assigns_role_from_overlapping_detection_box(self):
        tracker = RoleTracker(min_overlap=0.5)
        person = make_person(track_id=7, x1=100, y1=100, x2=200, y2=300)
        roles = tracker.assign([person], {"pitcher": (95, 95, 205, 305)})
        assert roles == {7: "pitcher"}

    def test_person_far_from_every_box_gets_no_role(self):
        tracker = RoleTracker(min_overlap=0.5)
        person = make_person(track_id=7, x1=1000, y1=1000, x2=1100, y2=1200)
        assert tracker.assign([person], {"pitcher": (95, 95, 205, 305)}) == {}

    def test_role_is_carried_forward_when_boxes_are_absent(self):
        # 役割bboxは stride のせいで毎フレーム来ない。一度決めたIDは維持する
        tracker = RoleTracker(min_overlap=0.5)
        person = make_person(track_id=7, x1=100, y1=100, x2=200, y2=300)
        tracker.assign([person], {"pitcher": (95, 95, 205, 305)})

        moved = make_person(track_id=7, x1=130, y1=110, x2=230, y2=310)
        assert tracker.assign([moved], {}) == {7: "pitcher"}

    def test_untracked_person_is_not_carried_forward(self):
        tracker = RoleTracker(min_overlap=0.5)
        person = make_person(track_id=None, x1=100, y1=100, x2=200, y2=300)
        assert tracker.assign([person], {}) == {}

    def test_two_people_get_distinct_roles(self):
        tracker = RoleTracker(min_overlap=0.5)
        pitcher = make_person(track_id=1, x1=100, y1=100, x2=200, y2=300)
        batter = make_person(track_id=2, x1=600, y1=150, x2=680, y2=350)
        roles = tracker.assign(
            [pitcher, batter],
            {"pitcher": (95, 95, 205, 305), "batter": (595, 145, 685, 355)},
        )
        assert roles == {1: "pitcher", 2: "batter"}

    def test_a_role_is_not_given_to_two_people(self):
        tracker = RoleTracker(min_overlap=0.5)
        best = make_person(track_id=1, x1=100, y1=100, x2=200, y2=300)
        worse = make_person(track_id=2, x1=150, y1=150, x2=260, y2=360)
        roles = tracker.assign([best, worse], {"pitcher": (95, 95, 205, 305)})
        assert list(roles.values()).count("pitcher") == 1

    def test_reset_forgets_assignments(self):
        tracker = RoleTracker(min_overlap=0.5)
        person = make_person(track_id=7, x1=100, y1=100, x2=200, y2=300)
        tracker.assign([person], {"pitcher": (95, 95, 205, 305)})
        tracker.reset()
        assert tracker.assign([person], {}) == {}


class TestPosePrefilterConfig:
    def test_trims_the_tail_and_keeps_the_windup_head(self):
        base = PrefilterConfig(pre_margin_sec=1.0, post_margin_sec=2.5)
        posed = pose_prefilter_config(base, pre_margin_sec=1.5, post_margin_sec=0.5)
        assert posed.pre_margin_sec == 1.5
        assert posed.post_margin_sec == 0.5

    def test_infers_every_frame_in_the_window(self):
        # 骨格は連続していないと動作解析に使えないので間引かない
        posed = pose_prefilter_config(PrefilterConfig())
        assert posed.search_stride == 1

    def test_roi_can_be_narrowed_for_window_derivation_only(self):
        posed = pose_prefilter_config(PrefilterConfig(), roi=(0.55, 0.2, 0.3, 0.45))
        assert posed.roi == (0.55, 0.2, 0.3, 0.45)

    def test_base_config_is_not_mutated(self):
        base = PrefilterConfig(pre_margin_sec=1.0)
        pose_prefilter_config(base, pre_margin_sec=9.0)
        assert base.pre_margin_sec == 1.0


class TestSkeletonEdges:
    def test_edges_reference_valid_keypoints(self):
        for a, b in SKELETON_EDGES:
            assert 0 <= a < len(KEYPOINT_NAMES)
            assert 0 <= b < len(KEYPOINT_NAMES)

    def test_coco_has_seventeen_keypoints(self):
        assert len(KEYPOINT_NAMES) == 17


class TestDrawSkeleton:
    def test_draws_onto_the_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        person = make_person(track_id=1, x1=100, y1=100, x2=300, y2=400)
        out = draw_skeleton(frame, person, color=(0, 255, 255))
        assert out.any(), "骨格が1ピクセルも描かれていない"

    def test_does_not_mutate_the_input_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        person = make_person(track_id=1, x1=100, y1=100, x2=300, y2=400)
        draw_skeleton(frame, person, color=(0, 255, 255))
        assert not frame.any(), "入力フレームが破壊的に変更された"

    def test_low_confidence_keypoints_are_skipped(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        person = make_person(track_id=1, x1=100, y1=100, x2=300, y2=400, conf=0.05)
        out = draw_skeleton(frame, person, color=(0, 255, 255), min_score=0.5)
        assert not out.any(), "低信頼のキーポイントが描かれている"


class TestSelectPerson:
    """対象の役割の人物を1人選ぶ。track.py の書き出しとアダプタで共有する。"""

    def test_role_assignment_wins_over_size(self):
        small = make_person(track_id=1, x1=0, y1=0, x2=20, y2=40)
        large = make_person(track_id=2, x1=0, y1=0, x2=200, y2=400)
        result = {"persons": [small, large], "roles": {1: "pitcher", 2: "catcher"}}

        person, mode = select_person(result, "pitcher", min_score=0.5)

        assert person is small
        assert mode == "role_bbox"

    def test_largest_skeleton_is_the_fallback(self):
        small = make_person(track_id=1, x1=0, y1=0, x2=20, y2=40)
        large = make_person(track_id=2, x1=0, y1=0, x2=200, y2=400)

        person, mode = select_person({"persons": [small, large], "roles": {}}, "pitcher", 0.5)

        assert person is large, "役割が決まらないときは最大の骨格を採る"
        assert mode == "largest_bbox"

    def test_other_roles_can_be_selected(self):
        pitcher = make_person(track_id=1, x1=0, y1=0, x2=20, y2=40)
        batter = make_person(track_id=2, x1=0, y1=0, x2=30, y2=60)
        result = {"persons": [pitcher, batter], "roles": {1: "pitcher", 2: "batter"}}

        person, mode = select_person(result, "batter", min_score=0.5)

        assert person is batter
        assert mode == "role_bbox"

    def test_no_person_is_reported(self):
        person, mode = select_person({"persons": [], "roles": {}}, "pitcher", 0.5)

        assert person is None
        assert mode == "none"

    def test_low_confidence_only_person_is_not_selected(self):
        """信頼できるキーポイントが1つも無い人物は、大きさを測れないので選ばない。"""
        faint = make_person(track_id=1, x1=0, y1=0, x2=200, y2=400, conf=0.05)

        person, mode = select_person({"persons": [faint], "roles": {}}, "pitcher", 0.5)

        assert person is None
        assert mode == "none"

    def test_person_without_track_id_can_still_be_the_fallback(self):
        """トラッカーIDが付かなくても、最大の骨格としてなら選ばれる。"""
        anonymous = make_person(track_id=None, x1=0, y1=0, x2=200, y2=400)

        person, mode = select_person({"persons": [anonymous], "roles": {}}, "pitcher", 0.5)

        assert person is anonymous
        assert mode == "largest_bbox"
