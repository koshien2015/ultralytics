from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="キャップ野球 投球分析",
    page_icon="⚾",
    layout="wide",
)

# ---------------------------------------------------------------------------
# データ準備ヘルパー
# ---------------------------------------------------------------------------

def pitches_to_df(pitches: List[Dict]) -> pd.DataFrame:
    rows = []
    for p in pitches:
        for pt in p["trajectory"]:
            rows.append({
                "pitch_id": p["pitch_id"],
                "is_strike": p["is_strike"],
                "release_frame": p["release_frame"],
                "frame": pt["frame"],
                "time": pt["time"],
                "x": pt["x"],
                "y": pt["y"],
                "source": pt["source"],
            })
    return pd.DataFrame(rows)


def pitcher_metrics_to_df(pitcher: List[Dict]) -> pd.DataFrame:
    rows = []
    for pm in pitcher:
        for f in pm["frames"]:
            rows.append({
                "pitch_id": pm["pitch_id"],
                "frame": f["frame"],
                "elbow_angle_deg": f["elbow_angle_deg"],
                "shoulder_tilt_deg": f["shoulder_tilt_deg"],
                "hip_angle_deg": f["hip_angle_deg"],
                "front_knee_angle_deg": f["front_knee_angle_deg"],
                "wrist_x": f["wrist_x"],
                "wrist_y": f["wrist_y"],
            })
    return pd.DataFrame(rows)


def pitcher_summary_df(pitcher: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "投球": f"投球 {pm['pitch_id']}",
            "リリースフレーム": pm["release_frame"],
            "肘角度 (deg)": pm["release_elbow_angle_deg"],
            "腰回転幅 (deg)": pm["hip_rotation_range_deg"],
            "手首X (px)": pm["release_wrist_x"],
            "手首Y (px)": pm["release_wrist_y"],
        }
        for pm in pitcher
    ])


def batter_summary_df(batter: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "投球": f"投球 {bm['pitch_id']}",
            "スイング開始": bm["swing_start_frame"],
            "スイング終了": bm["swing_end_frame"],
            "腰回転幅 (deg)": bm["hip_rotation_range_deg"],
            "肩水平差 avg (px)": bm["avg_shoulder_level_diff_px"],
            "頭の移動 (px)": bm["head_displacement_px"],
        }
        for bm in batter
    ])


# ---------------------------------------------------------------------------
# サイドバー
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("データ読み込み")
    json_file = st.file_uploader("pitches.json（軌跡）", type="json", key="traj")
    pose_file = st.file_uploader("pose_analysis.json（姿勢）", type="json", key="pose")
    video_file = st.file_uploader("result.mp4（任意）", type=["mp4", "mov"])
    st.divider()
    st.caption("パイプラインの output/ から各ファイルを選択してください。")

# ---------------------------------------------------------------------------
# データ読み込み
# ---------------------------------------------------------------------------

traj_data: Optional[Dict] = json.load(json_file) if json_file else None
pose_data: Optional[Dict] = json.load(pose_file) if pose_file else None

pitches = traj_data["pitches"] if traj_data else []
df = pitches_to_df(pitches) if pitches else pd.DataFrame()
meta = traj_data.get("metadata", {}) if traj_data else {}

pitcher_list = pose_data.get("pitcher", []) if pose_data else []
batter_list = pose_data.get("batter", []) if pose_data else []

if not traj_data and not pose_data:
    st.info("サイドバーから JSON ファイルを読み込んでください。")
    st.stop()

# ---------------------------------------------------------------------------
# タブ
# ---------------------------------------------------------------------------

tab_overview, tab_trajectory, tab_pitcher, tab_batter, tab_video = st.tabs(
    ["📊 概要", "🎯 軌跡分析", "⚾ 投手分析", "🏏 打者分析", "🎬 動画"]
)

# ── 概要 ───────────────────────────────────────────────────────────────────

with tab_overview:
    st.subheader("投球サマリー")

    strike_count = sum(1 for p in pitches if p.get("is_strike") is True)
    ball_count = sum(1 for p in pitches if p.get("is_strike") is False)
    unknown_count = sum(1 for p in pitches if p.get("is_strike") is None)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("総投球数", len(pitches))
    col2.metric("ストライク", strike_count)
    col3.metric("ボール", ball_count)
    col4.metric("未判定", unknown_count)

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**メタデータ**")
        st.json({
            "動画": meta.get("video_file", "—"),
            "FPS": meta.get("fps", "—"),
            "カメラ角度": meta.get("camera_angle", "—"),
        })

    with col_b:
        if not df.empty:
            st.markdown("**検出ソース内訳**")
            src = df.groupby("source")["frame"].count().reset_index()
            src.columns = ["source", "count"]
            fig = px.pie(src, names="source", values="count",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=250)
            st.plotly_chart(fig, use_container_width=True)

# ── 軌跡分析 ───────────────────────────────────────────────────────────────

with tab_trajectory:
    if df.empty:
        st.info("pitches.json を読み込んでください。")
    else:
        col_l, col_r = st.columns([1, 2])
        with col_l:
            show_all = st.checkbox("全投球を重ねて表示", value=True)
            pitch_ids = sorted(df["pitch_id"].unique())
            selected = st.selectbox("投球を選択", pitch_ids,
                                    format_func=lambda x: f"投球 {x}",
                                    disabled=show_all)
            color_src = st.checkbox("検出ソースで色分け", value=True)

        with col_r:
            plot_df = df if show_all else df[df["pitch_id"] == selected]
            fig = go.Figure()
            group_col = "source" if color_src else "pitch_id"
            for key, grp in plot_df.groupby(group_col):
                for pid, sub in (grp.groupby("pitch_id") if color_src else [(key, grp)]):
                    sub_s = sub.sort_values("frame")
                    fig.add_trace(go.Scatter(
                        x=sub_s["x"], y=sub_s["y"],
                        mode="lines+markers",
                        name=f"#{pid} {key}" if color_src else f"投球 {key}",
                        marker=dict(size=5), line=dict(width=2),
                    ))
            fig.update_layout(
                title="ボール軌跡",
                xaxis_title="X（正規化）", yaxis_title="Y（正規化）",
                yaxis=dict(autorange="reversed"),
                height=430, margin=dict(t=40, b=60),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25),
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("生データ"):
            st.dataframe(df, use_container_width=True)

# ── 投手分析 ───────────────────────────────────────────────────────────────

with tab_pitcher:
    if not pitcher_list:
        st.info("pose_analysis.json を読み込んでください。")
    else:
        pm_df = pitcher_metrics_to_df(pitcher_list)
        pitch_ids = sorted(pm_df["pitch_id"].unique())

        st.subheader("投球ごとのサマリー")
        st.dataframe(pitcher_summary_df(pitcher_list), use_container_width=True)

        st.divider()
        st.subheader("モーション中の角度推移")

        col_l, col_r = st.columns([1, 3])
        with col_l:
            compare_mode = st.checkbox("全投球を重ねて比較", value=False)
            sel_pitch = st.selectbox("投球を選択", pitch_ids,
                                     format_func=lambda x: f"投球 {x}",
                                     disabled=compare_mode)
            metrics = st.multiselect(
                "表示する指標",
                ["elbow_angle_deg", "shoulder_tilt_deg", "hip_angle_deg", "front_knee_angle_deg"],
                default=["elbow_angle_deg", "hip_angle_deg"],
                format_func=lambda x: {
                    "elbow_angle_deg": "肘角度",
                    "shoulder_tilt_deg": "肩傾き",
                    "hip_angle_deg": "腰回転角",
                    "front_knee_angle_deg": "前膝屈曲角",
                }[x],
            )

        with col_r:
            plot_df = pm_df if compare_mode else pm_df[pm_df["pitch_id"] == sel_pitch]
            for metric in metrics:
                label = {
                    "elbow_angle_deg": "肘角度 (deg)",
                    "shoulder_tilt_deg": "肩傾き (px)",
                    "hip_angle_deg": "腰回転角 (deg)",
                    "front_knee_angle_deg": "前膝屈曲角 (deg)",
                }[metric]
                fig = px.line(
                    plot_df.dropna(subset=[metric]).sort_values("frame"),
                    x="frame", y=metric,
                    color="pitch_id" if compare_mode else None,
                    labels={"frame": "フレーム", metric: label, "pitch_id": "投球"},
                    title=label,
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                )
                fig.update_layout(height=280, margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("リリースポイント（手首座標）")
        release_pts = [
            {"pitch_id": pm["pitch_id"], "x": pm["release_wrist_x"], "y": pm["release_wrist_y"]}
            for pm in pitcher_list
            if pm["release_wrist_x"] is not None
        ]
        if release_pts:
            rp_df = pd.DataFrame(release_pts)
            fig_rp = px.scatter(
                rp_df, x="x", y="y", text="pitch_id",
                labels={"x": "X (px)", "y": "Y (px)", "pitch_id": "投球"},
                title="リリース時の手首位置（投球間比較）",
            )
            fig_rp.update_traces(textposition="top center", marker=dict(size=12))
            fig_rp.update_layout(yaxis=dict(autorange="reversed"), height=350)
            st.plotly_chart(fig_rp, use_container_width=True)

# ── 打者分析 ───────────────────────────────────────────────────────────────

with tab_batter:
    if not batter_list:
        st.info("pose_analysis.json を読み込んでください。")
    else:
        st.subheader("スイングごとのサマリー")
        st.dataframe(batter_summary_df(batter_list), use_container_width=True)

        st.divider()
        col_l, col_r = st.columns([1, 3])

        batter_pitch_ids = [bm["pitch_id"] for bm in batter_list]
        with col_l:
            compare_sw = st.checkbox("全スイングを重ねて表示", value=False)
            sel_batter = st.selectbox("スイングを選択", batter_pitch_ids,
                                      format_func=lambda x: f"投球 {x}",
                                      disabled=compare_sw)

        with col_r:
            # スイングパス
            st.markdown("**スイングパス（手首軌跡）**")
            fig_sw = go.Figure()
            targets = batter_list if compare_sw else [b for b in batter_list if b["pitch_id"] == sel_batter]
            for bm in targets:
                path = bm["wrist_path"]
                if path:
                    fig_sw.add_trace(go.Scatter(
                        x=[p["x"] for p in path],
                        y=[p["y"] for p in path],
                        mode="lines+markers",
                        name=f"投球 {bm['pitch_id']}",
                        marker=dict(size=6),
                        line=dict(width=2),
                    ))
            fig_sw.update_layout(
                xaxis_title="X (px)", yaxis_title="Y (px)",
                yaxis=dict(autorange="reversed"),
                height=350, margin=dict(t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25),
            )
            st.plotly_chart(fig_sw, use_container_width=True)

            # 姿勢チェック（腰回転・肩水平度の時系列）
            st.markdown("**スイング中の姿勢（フレーム推移）**")
            bm_rows = []
            for bm in targets:
                for f in bm["frames"]:
                    bm_rows.append({
                        "pitch_id": bm["pitch_id"],
                        "frame": f["frame"],
                        "hip_angle_deg": f["hip_angle_deg"],
                        "shoulder_level_diff_px": f["shoulder_level_diff_px"],
                        "front_knee_angle_deg": f["front_knee_angle_deg"],
                    })
            if bm_rows:
                bm_frame_df = pd.DataFrame(bm_rows)
                for col_name, label in [
                    ("hip_angle_deg", "腰回転角 (deg)"),
                    ("shoulder_level_diff_px", "肩水平差 (px)"),
                    ("front_knee_angle_deg", "前膝屈曲角 (deg)"),
                ]:
                    sub = bm_frame_df.dropna(subset=[col_name]).sort_values("frame")
                    if sub.empty:
                        continue
                    fig_b = px.line(
                        sub, x="frame", y=col_name,
                        color="pitch_id" if compare_sw else None,
                        labels={"frame": "フレーム", col_name: label, "pitch_id": "投球"},
                        title=label,
                    )
                    fig_b.update_layout(height=250, margin=dict(t=40, b=10))
                    st.plotly_chart(fig_b, use_container_width=True)

# ── 動画 ───────────────────────────────────────────────────────────────────

with tab_video:
    if video_file:
        st.video(video_file)
        st.caption("ネオン軌跡描画済み動画")
    else:
        st.info("サイドバーから result.mp4 を読み込むと表示されます。")
