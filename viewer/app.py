from __future__ import annotations

import json
from pathlib import Path
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
# データロード
# ---------------------------------------------------------------------------

def load_pitches(json_path: Path) -> Dict[str, Any]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


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


# ---------------------------------------------------------------------------
# サイドバー
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("データ読み込み")
    json_file = st.file_uploader("pitches.json", type="json")
    video_file = st.file_uploader("result.mp4（任意）", type=["mp4", "mov"])

    st.divider()
    st.caption("pitching パイプラインの出力ディレクトリから\n各ファイルを選択してください。")

if json_file is None:
    st.info("サイドバーから pitches.json を読み込んでください。")
    st.stop()

# ---------------------------------------------------------------------------
# データ準備
# ---------------------------------------------------------------------------

data = json.load(json_file)
meta = data.get("metadata", {})
pitches = data.get("pitches", [])
df = pitches_to_df(pitches) if pitches else pd.DataFrame()

pitch_ids = [p["pitch_id"] for p in pitches]
strike_count = sum(1 for p in pitches if p.get("is_strike") is True)
ball_count = sum(1 for p in pitches if p.get("is_strike") is False)
unknown_count = sum(1 for p in pitches if p.get("is_strike") is None)

# ---------------------------------------------------------------------------
# タブ
# ---------------------------------------------------------------------------

tab_overview, tab_trajectory, tab_video, tab_pose = st.tabs(
    ["📊 概要", "🎯 軌跡分析", "🎬 動画", "🏃 姿勢分析"]
)

# ── 概要 ───────────────────────────────────────────────────────────────────

with tab_overview:
    st.subheader("投球サマリー")

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
            "動画ファイル": meta.get("video_file", "—"),
            "FPS": meta.get("fps", "—"),
            "カメラ角度 (deg)": meta.get("camera_angle", "—"),
        })

    with col_b:
        if not df.empty:
            st.markdown("**検出ソース内訳**")
            src_counts = df.groupby("source")["frame"].count().reset_index()
            src_counts.columns = ["source", "count"]
            fig_src = px.pie(
                src_counts, names="source", values="count",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_src.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=260)
            st.plotly_chart(fig_src, use_container_width=True)

    if not df.empty:
        st.divider()
        st.markdown("**投球ごとの軌跡ポイント数**")
        pt_counts = df.groupby("pitch_id")["frame"].count().reset_index()
        pt_counts.columns = ["pitch_id", "points"]
        pt_counts["pitch_id"] = pt_counts["pitch_id"].astype(str)
        fig_pts = px.bar(
            pt_counts, x="pitch_id", y="points",
            labels={"pitch_id": "投球 ID", "points": "検出ポイント数"},
            color_discrete_sequence=["#00b4d8"],
        )
        fig_pts.update_layout(height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig_pts, use_container_width=True)

# ── 軌跡分析 ───────────────────────────────────────────────────────────────

with tab_trajectory:
    if df.empty:
        st.info("投球データがありません。")
    else:
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown("**表示設定**")
            show_all = st.checkbox("全投球を重ねて表示", value=True)
            selected_id = st.selectbox(
                "投球を選択（単体表示）",
                options=pitch_ids,
                format_func=lambda x: f"投球 {x}",
                disabled=show_all,
            )
            color_by_source = st.checkbox("検出ソースで色分け", value=True)

        with col_right:
            if show_all:
                plot_df = df.copy()
                color_col = "source" if color_by_source else "pitch_id"
                title = "全投球の軌跡"
            else:
                plot_df = df[df["pitch_id"] == selected_id].copy()
                color_col = "source" if color_by_source else None
                title = f"投球 {selected_id} の軌跡"

            fig_traj = go.Figure()

            if color_by_source:
                for src, grp in plot_df.groupby("source"):
                    for pid, sub in grp.groupby("pitch_id"):
                        sub_sorted = sub.sort_values("frame")
                        fig_traj.add_trace(go.Scatter(
                            x=sub_sorted["x"], y=sub_sorted["y"],
                            mode="lines+markers",
                            name=f"#{pid} {src}",
                            marker=dict(size=5),
                            line=dict(width=2),
                        ))
            else:
                for pid, grp in plot_df.groupby("pitch_id"):
                    grp_sorted = grp.sort_values("frame")
                    fig_traj.add_trace(go.Scatter(
                        x=grp_sorted["x"], y=grp_sorted["y"],
                        mode="lines+markers",
                        name=f"投球 {pid}",
                        marker=dict(size=5),
                        line=dict(width=2),
                    ))

            fig_traj.update_layout(
                title=title,
                xaxis_title="X（正規化）",
                yaxis_title="Y（正規化）",
                yaxis=dict(autorange="reversed"),  # 画像座標系（上が0）
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                margin=dict(t=40, b=80),
            )
            st.plotly_chart(fig_traj, use_container_width=True)

        # 時系列グラフ
        st.divider()
        st.markdown("**時系列（X / Y 座標）**")
        single_df = df if show_all else df[df["pitch_id"] == selected_id]
        single_df = single_df.sort_values(["pitch_id", "frame"])

        fig_ts = px.line(
            single_df, x="time", y=["x", "y"],
            color_discrete_map={"x": "#00b4d8", "y": "#f77f00"},
            labels={"time": "経過時間 (s)", "value": "座標（正規化）", "variable": "軸"},
            facet_col="variable" if show_all else None,
        )
        fig_ts.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig_ts, use_container_width=True)

        # 生データテーブル
        with st.expander("生データ（DataFrame）"):
            st.dataframe(df, use_container_width=True)

# ── 動画 ───────────────────────────────────────────────────────────────────

with tab_video:
    if video_file is not None:
        st.video(video_file)
        st.caption("ネオン軌跡描画済み動画（パイプライン出力）")
    else:
        st.info("サイドバーから result.mp4 を読み込むと動画が表示されます。")

# ── 姿勢分析 ───────────────────────────────────────────────────────────────

with tab_pose:
    st.info(
        "姿勢分析機能は現在実装中です。\n\n"
        "追加予定の分析項目：\n"
        "- 投手: リリースポイント・腕の角度・ストライド長・腰回転角度\n"
        "- 打者: スイング検出・スタンス分類・インパクト判定・体重移動\n"
        "- 投打統合: 反応時間・ボール通過点 vs スイング軌跡"
    )

    st.markdown("### 将来の表示イメージ")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**投手メトリクス（投球ごと）**")
        dummy_pitcher = pd.DataFrame({
            "投球": [f"投球 {i}" for i in range(1, 6)],
            "腕の角度 (deg)": [None] * 5,
            "ストライド長 (px)": [None] * 5,
            "腰回転 (deg)": [None] * 5,
        })
        st.dataframe(dummy_pitcher, use_container_width=True)

    with col2:
        st.markdown("**打者メトリクス（投球ごと）**")
        dummy_batter = pd.DataFrame({
            "投球": [f"投球 {i}" for i in range(1, 6)],
            "スイング": [None] * 5,
            "スタンス": [None] * 5,
            "反応時間 (s)": [None] * 5,
        })
        st.dataframe(dummy_batter, use_container_width=True)
