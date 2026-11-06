"""
軌跡フィッティング・後処理モジュール

YOLOのフレーム単位検出は欠損・誤検出を含むため、
投球1球分の検出点列に対して物理モデル（時間の2次多項式）を
RANSACでフィットし、以下を行う:

- 誤検出（軌跡に乗らない点）の除去
- 欠損フレームの補間
- 通過点（コース）のサブフレーム精度での推定
- 平均球速の推定

座標系は PitchingAnalyzer の正規化座標を前提とする:
- x: ストライクゾーン中心=0、ゾーン端=±0.5
- y: ゾーン下端=0、ゾーン上端=1
- t: リリースからの経過時間（秒）
"""

import random
from dataclasses import dataclass

import numpy as np

# フィットに最低限必要な点数（2次式の自由度3 + 冗長性）
MIN_POINTS_FOR_FIT = 5
# RANSACのデフォルト設定
DEFAULT_RANSAC_ITERATIONS = 200
DEFAULT_INLIER_THRESHOLD = 0.15  # 正規化座標系での残差許容量
# 投本間距離（キャップ野球規定）
PITCH_DISTANCE_M = 9.22


@dataclass(frozen=True)
class TrajectoryPoint:
    """1フレーム分の検出点"""
    frame: int
    t: float  # リリースからの経過時間（秒）
    x: float
    y: float


@dataclass(frozen=True)
class FitResult:
    """フィット結果（係数は [c0, c1, c2] で v = c0 + c1*t + c2*t^2）"""
    coeffs_x: tuple
    coeffs_y: tuple
    inliers: tuple  # TrajectoryPoint のタプル
    outliers: tuple  # TrajectoryPoint のタプル
    rmse: float


def _fit_quadratic(ts, vs):
    """時間の2次多項式を最小二乗フィットし係数 [c0, c1, c2] を返す"""
    design = np.stack([np.ones_like(ts), ts, ts ** 2], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, vs, rcond=None)
    return coeffs


def _evaluate(coeffs, t):
    """係数 [c0, c1, c2] を時刻 t で評価する"""
    return coeffs[0] + coeffs[1] * t + coeffs[2] * t * t


def _residuals(points, coeffs_x, coeffs_y):
    """各点の予測位置からのユークリッド残差"""
    ts = np.array([p.t for p in points])
    xs = np.array([p.x for p in points])
    ys = np.array([p.y for p in points])
    pred_x = _evaluate(coeffs_x, ts)
    pred_y = _evaluate(coeffs_y, ts)
    return np.sqrt((xs - pred_x) ** 2 + (ys - pred_y) ** 2)


def ransac_fit(
    points,
    inlier_threshold=DEFAULT_INLIER_THRESHOLD,
    iterations=DEFAULT_RANSAC_ITERATIONS,
    seed=0,
):
    """
    検出点列にRANSACで2次軌跡をフィットする

    Args:
        points: TrajectoryPoint のリスト
        inlier_threshold: インライア判定の残差閾値（正規化座標）
        iterations: RANSACの試行回数
        seed: 乱数シード（結果を再現可能にするため固定）

    Returns:
        FitResult

    Raises:
        ValueError: 点数が不足している場合
    """
    if len(points) < MIN_POINTS_FOR_FIT:
        raise ValueError(
            f"フィットには最低{MIN_POINTS_FOR_FIT}点必要です（入力: {len(points)}点）"
        )

    rng = random.Random(seed)
    best_inlier_mask = None
    best_inlier_count = 0

    ts_all = np.array([p.t for p in points])
    if np.ptp(ts_all) <= 0:
        raise ValueError("全ての点が同一時刻です。時系列データを渡してください")

    for _ in range(iterations):
        sample = rng.sample(points, 3)
        sample_ts = np.array([p.t for p in sample])
        # 同一時刻の点が混ざるとランク落ちするためスキップ
        if len(set(sample_ts.tolist())) < 3:
            continue

        coeffs_x = _fit_quadratic(sample_ts, np.array([p.x for p in sample]))
        coeffs_y = _fit_quadratic(sample_ts, np.array([p.y for p in sample]))

        residuals = _residuals(points, coeffs_x, coeffs_y)
        inlier_mask = residuals < inlier_threshold
        inlier_count = int(inlier_mask.sum())

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask

    if best_inlier_mask is None or best_inlier_count < MIN_POINTS_FOR_FIT:
        raise ValueError(
            f"軌跡に乗る点が{MIN_POINTS_FOR_FIT}点未満でした"
            f"（最良: {best_inlier_count}点）。誤検出が支配的か、閾値が厳しすぎます"
        )

    # インライア全体で最終フィット（精度を上げる）
    inlier_points = [p for p, ok in zip(points, best_inlier_mask) if ok]
    outlier_points = [p for p, ok in zip(points, best_inlier_mask) if not ok]

    inlier_ts = np.array([p.t for p in inlier_points])
    coeffs_x = _fit_quadratic(inlier_ts, np.array([p.x for p in inlier_points]))
    coeffs_y = _fit_quadratic(inlier_ts, np.array([p.y for p in inlier_points]))

    rmse = float(np.sqrt(np.mean(_residuals(inlier_points, coeffs_x, coeffs_y) ** 2)))

    return FitResult(
        coeffs_x=tuple(float(c) for c in coeffs_x),
        coeffs_y=tuple(float(c) for c in coeffs_y),
        inliers=tuple(inlier_points),
        outliers=tuple(outlier_points),
        rmse=rmse,
    )


def position_at(fit, t):
    """フィット済み軌跡上の時刻 t の位置 (x, y) を返す"""
    return (
        float(_evaluate(np.array(fit.coeffs_x), t)),
        float(_evaluate(np.array(fit.coeffs_y), t)),
    )


def interpolate_gaps(fit, fps):
    """
    インライアの時間範囲内で欠損フレームを補間した点列を返す

    Args:
        fit: FitResult
        fps: 動画のフレームレート

    Returns:
        TrajectoryPoint のリスト（実測フレーム + 補間フレーム、時刻順）
    """
    if fps <= 0:
        raise ValueError(f"fpsは正の値が必要です（入力: {fps}）")

    detected_frames = {p.frame for p in fit.inliers}
    first = min(fit.inliers, key=lambda p: p.frame)
    last = max(fit.inliers, key=lambda p: p.frame)

    completed = list(fit.inliers)
    for frame in range(first.frame + 1, last.frame):
        if frame in detected_frames:
            continue
        t = first.t + (frame - first.frame) / fps
        x, y = position_at(fit, t)
        completed.append(TrajectoryPoint(frame=frame, t=t, x=x, y=y))

    return sorted(completed, key=lambda p: p.frame)


def zone_from_position(x_norm, y_norm):
    """
    正規化座標から9分割コースを判定する

    ゾーン番号は捕手（カメラ）視点で
        1 2 3   （上段: 左・中・右）
        4 5 6
        7 8 9   （下段）
    ゾーン外は 'ball' を返す。

    Args:
        x_norm: ゾーン中心=0、ゾーン端=±0.5
        y_norm: ゾーン下端=0、ゾーン上端=1

    Returns:
        1〜9 の int、またはゾーン外なら文字列 'ball'
    """
    if not (-0.5 <= x_norm <= 0.5) or not (0.0 <= y_norm <= 1.0):
        return "ball"

    # 列: 左(0)・中(1)・右(2)
    col = min(int((x_norm + 0.5) * 3), 2)
    # 行: 上(0)・中(1)・下(2) — y_normは上が1なので反転
    row = min(int((1.0 - y_norm) * 3), 2)

    return row * 3 + col + 1


def estimate_speed_kmh(fit, distance_m=PITCH_DISTANCE_M):
    """
    リリース〜捕球の平均球速を推定する（km/h）

    単眼カメラでは奥行きが取れないため、インライアの時間幅が
    投本間距離の移動時間に相当すると仮定した平均値。
    """
    ts = [p.t for p in fit.inliers]
    duration = max(ts) - min(ts)
    if duration <= 0:
        raise ValueError("軌跡の時間幅が0です。球速を推定できません")
    return float(distance_m / duration * 3.6)


def analyze_pitch(trajectory, fps, inlier_threshold=DEFAULT_INLIER_THRESHOLD):
    """
    1球分の軌跡データ（PitchingAnalyzerのJSON形式）を解析する

    Args:
        trajectory: [{'frame': int, 'time': float, 'x': float, 'y': float}, ...]
        fps: 動画のフレームレート
        inlier_threshold: RANSACのインライア閾値

    Returns:
        dict: {
            'course_zone': 1-9 または 'ball',
            'course_x': 通過点x（正規化）,
            'course_y': 通過点y（正規化）,
            'speed_kmh': 平均球速,
            'num_detected': 実検出点数,
            'num_inliers': 軌跡に採用された点数,
            'num_outliers': 誤検出として除外された点数,
            'fit_rmse': フィット残差,
            'trajectory': 補間済み軌跡のリスト,
        }

    Raises:
        ValueError: 点数不足・フィット不能の場合
    """
    points = [
        TrajectoryPoint(frame=int(p["frame"]), t=float(p["time"]),
                        x=float(p["x"]), y=float(p["y"]))
        for p in trajectory
    ]

    fit = ransac_fit(points, inlier_threshold=inlier_threshold)

    # 通過点 = 軌跡の最終時刻（捕球直前）での位置
    t_end = max(p.t for p in fit.inliers)
    course_x, course_y = position_at(fit, t_end)

    completed = interpolate_gaps(fit, fps)

    return {
        "course_zone": zone_from_position(course_x, course_y),
        "course_x": course_x,
        "course_y": course_y,
        "speed_kmh": estimate_speed_kmh(fit),
        "num_detected": len(points),
        "num_inliers": len(fit.inliers),
        "num_outliers": len(fit.outliers),
        "fit_rmse": fit.rmse,
        "trajectory": [
            {"frame": p.frame, "time": p.t, "x": p.x, "y": p.y} for p in completed
        ],
    }
