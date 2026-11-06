"""追跡の運動モデル。最近傍追跡とカルマンフィルタで同じ追跡ループを共有する。"""

from dataclasses import dataclass, replace

import numpy as np

from .config import KalmanConfig, TrackerConfig


@dataclass(frozen=True)
class NearestState:
    x: float  # 最後の観測位置
    y: float
    vx: float
    vy: float
    t_obs: float
    t_now: float
    has_velocity: bool = False


class NearestMotion:
    """手法A: 移動予測位置に最も近い候補を、最大移動距離の範囲で対応付ける"""

    emits_prediction = False

    def __init__(self, config: TrackerConfig):
        self.max_speed = config.max_speed_px_s
        self.max_error = config.max_prediction_error_px

    def start(self, x, y, t):
        return NearestState(x, y, 0.0, 0.0, t, t)

    def advance(self, state, t):
        return replace(state, t_now=t)

    def position(self, state):
        elapsed = state.t_now - state.t_obs
        return state.x + state.vx * elapsed, state.y + state.vy * elapsed

    def score(self, state, x, y):
        px, py = self.position(state)
        distance = float(np.hypot(x - px, y - py))
        # 最後の観測から物理的に届かない候補は採らない
        reach = float(np.hypot(x - state.x, y - state.y))
        reachable = reach <= self.max_speed * (state.t_now - state.t_obs)
        # 欠損が続くと届く範囲は広がるので、速度が分かれば予測からの距離でも絞る
        return distance, reachable and (not state.has_velocity or distance <= self.max_error)

    def correct(self, state, x, y, t):
        elapsed = t - state.t_obs
        return NearestState(x, y, (x - state.x) / elapsed, (y - state.y) / elapsed, t, t, True)


@dataclass(frozen=True, eq=False)
class KalmanState:
    mean: np.ndarray  # [x, y, vx, vy]
    cov: np.ndarray
    t: float


OBSERVE = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


class KalmanMotion:
    """手法B: 等速運動モデルのカルマンフィルタ。マハラノビス距離で観測をゲートする"""

    emits_prediction = True

    def __init__(self, config: KalmanConfig):
        self.config = config
        self.obs_cov = np.eye(2) * config.obs_noise_px**2

    def start(self, x, y, t):
        pos_var = self.config.obs_noise_px**2
        vel_var = self.config.init_speed_px_s**2
        return KalmanState(
            np.array([x, y, 0.0, 0.0]), np.diag([pos_var, pos_var, vel_var, vel_var]), t
        )

    def _transition(self, dt):
        move = np.eye(4)
        move[0, 2] = move[1, 3] = dt
        var = self.config.accel_noise_px_s2**2
        q_pos, q_cross, q_vel = var * dt**4 / 4, var * dt**3 / 2, var * dt**2
        noise = np.array(
            [
                [q_pos, 0.0, q_cross, 0.0],
                [0.0, q_pos, 0.0, q_cross],
                [q_cross, 0.0, q_vel, 0.0],
                [0.0, q_cross, 0.0, q_vel],
            ]
        )
        return move, noise

    def advance(self, state, t):
        move, noise = self._transition(t - state.t)
        return KalmanState(move @ state.mean, move @ state.cov @ move.T + noise, t)

    def position(self, state):
        return float(state.mean[0]), float(state.mean[1])

    def _innovation(self, state, x, y):
        residual = np.array([x, y]) - OBSERVE @ state.mean
        return residual, OBSERVE @ state.cov @ OBSERVE.T + self.obs_cov

    def score(self, state, x, y):
        residual, covariance = self._innovation(state, x, y)
        squared = float(residual @ np.linalg.solve(covariance, residual))
        return squared, squared <= self.config.gate_chi2

    def correct(self, state, x, y, t):
        residual, covariance = self._innovation(state, x, y)
        gain = state.cov @ OBSERVE.T @ np.linalg.inv(covariance)
        return KalmanState(
            state.mean + gain @ residual, (np.eye(4) - gain @ OBSERVE) @ state.cov, t
        )
