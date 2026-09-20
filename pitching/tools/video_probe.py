#!/usr/bin/env python3
"""動画ファイル自身から時間軸と寸法を読む。**自己申告を使わない。**

`metadata.json` の値ではなく、保存されたファイルの中身を見る。理由は 2 つ。

1. **アプリ外で撮った動画には `metadata.json` が無い。** レーダーの無いクリップを
   同じビューアで見る以上、動画だけから始められなければならない。
2. **自己申告と実体はずれる。** 実データで `durationMs` は 57359（metadata）に対して
   57364（ファイル）、fps は 239.48835806346878 に対して 239.4883546942585 だった。
   2 台の PC が同じ指紋を出すには、材料の出所を 1 つに決めるしかない。

**時刻の正本は `stts` の目盛表であって平均 fps ではない。**
実データの `stts` は 100 目盛 × 10803 枚 と 101 目盛 × 2935 枚 に割れている
（実測 fps が整数でないため）。平均で割り戻すと、指示書が禁じている
`fps × 秒` に戻ってしまい、長い動画の終端でフレームがずれる。
目盛表はランレングスのままなら極小（実データで 2 区間）なので、そのまま持ち回れる。

QuickTime(.mov) と ISO BMFF(.mp4) は箱の構造がほぼ同じなので、同じ読み方で両方扱える。
"""

from __future__ import annotations

import struct
from pathlib import Path

# 容れ物として中へ降りる箱。mdat は降りない（中身は画素で、偽の一致を拾う）。
CONTAINERS = (b"moov", b"trak", b"mdia", b"minf", b"stbl")

# ブラウザが受け付けるコンテナの申告（ftyp の major brand）。
#
# **コーデックとは独立した 2 つ目の壁。** 実データの元動画は `qt  `（QuickTime）で、
# Chrome / Edge / Firefox は ISO-BMFF しか受けない。中身が H.264 でも弾かれる。
# 読み取り自体はブランドを見ずに行う（構造はほぼ同じなので読めてしまう）。
# 見るのは「ビューアで再生できるか」を判断するときだけ。
BROWSER_BRANDS = ("isom", "iso2", "iso4", "iso5", "iso6", "mp41", "mp42", "avc1", "dash")


class ProbeError(RuntimeError):
    """動画から必要な情報を読めない。**推測で埋めない。**"""


def probe_video(path: str | Path) -> dict:
    """動画から時間軸と寸法を読む。

    戻り値:
        path            読み元のパス（呼ぶ側が指紋を計算し直せるように）
        major_brand     ftyp の申告（"qt  " / "isom" など）
        timescale       1 秒あたりの目盛数
        frame_count     映像トラックのサンプル数
        duration_ticks  目盛で表した尺
        duration_ms     ミリ秒で表した尺
        fps             平均 fps。**表示用の要約値。時刻の計算には使わない**
        tick_table      [[枚数, 目盛], ...] のランレングス。**これが時刻の正本**
        width/height    回転を当てる前の符号化サイズ
        display_width/display_height  回転を当てたあとの見た目のサイズ
        rotation_deg    0 / 90 / 180 / 270
    """
    target = Path(path)
    try:
        handle = target.open("rb")
    except OSError as error:
        raise ProbeError(f"動画を開けない: {target}: {error}") from error

    with handle:
        handle.seek(0, 2)
        size = handle.tell()
        if size == 0:
            raise ProbeError(f"動画が空: {target}")
        brand = _major_brand(handle)
        moov = _find_moov(handle, size)
        if moov is None:
            raise ProbeError(
                f"moov が見つからない（QuickTime / MP4 として読めない）: {target}")
        handle.seek(moov[0])
        body = handle.read(moov[1] - moov[0])

    video = _video_track(body)
    if video is None:
        raise ProbeError(f"映像トラックが見つからない: {target}")

    timescale, duration_ticks, table, width, height, rotation = video
    frame_count = sum(count for count, _ in table)
    if timescale <= 0 or frame_count <= 0:
        raise ProbeError(f"時間情報が壊れている: {target}")

    # 尺は目盛表の合計から出す。mdhd の duration は編集リストの影響を受けうるので、
    # 実際に並んでいるフレームの合計の方が、フレーム番号と整合する。
    ticks = sum(count * delta for count, delta in table)
    duration_ms = ticks / timescale * 1000.0
    display = (height, width) if rotation in (90, 270) else (width, height)

    return {
        # 読み元。呼ぶ側が指紋を計算し直せるようにする。これが無いと
        # 「照合していない」のか「照合して一致した」のかを区別できない。
        "path": str(target),
        # コンテナの申告。ブラウザで再生できるかの判断に使う。
        "major_brand": brand,
        "timescale": timescale,
        "frame_count": frame_count,
        "duration_ticks": ticks,
        "mdhd_duration_ticks": duration_ticks,
        "duration_ms": duration_ms,
        # 平均。**時刻の計算には使わない。** 表示と指紋の材料にだけ使う。
        "fps": frame_count / (ticks / timescale) if ticks else 0.0,
        "tick_table": [[count, delta] for count, delta in table],
        "width": width,
        "height": height,
        "display_width": display[0],
        "display_height": display[1],
        "rotation_deg": rotation,
    }


def is_browser_playable_container(info: dict) -> bool:
    """このコンテナをブラウザが受け付けるか。

    コーデックは見ない（別の壁）。ここで見るのは ftyp の申告だけ。
    別 PC から戻ってきたクリップをフォルダへ組み込む時点で弾けるようにする——
    Windows の実機が無くても検出できる、数少ない Windows 固有の不具合。
    """
    return info.get("major_brand") in BROWSER_BRANDS


def _major_brand(handle) -> str | None:
    """ftyp の major brand を読む。無ければ None。"""
    handle.seek(0)
    head = handle.read(16)
    if len(head) < 16 or head[4:8] != b"ftyp":
        return None
    return head[8:12].decode("latin1", "replace")


def frame_to_ms(info: dict, frame: int) -> float:
    """フレーム番号 → 動画内の時刻（ms）。目盛表を積み上げる。

    `frame / fps` で出さない。平均 fps は区間ごとの差を潰すので、
    実データでは終端で 1ms 以上ずれる（テストで数値として残してある）。
    """
    count_total = info["frame_count"]
    if frame < 0 or frame >= count_total:
        raise ProbeError(f"フレーム {frame} は動画の範囲外（全 {count_total} 枚）")

    ticks = 0
    remaining = frame
    for count, delta in info["tick_table"]:
        if remaining < count:
            ticks += remaining * delta
            break
        ticks += count * delta
        remaining -= count
    return ticks / info["timescale"] * 1000.0


def ms_to_frame(info: dict, ms: float) -> int:
    """動画内の時刻（ms）→ そのとき表示されているフレーム番号。

    境界では「その時刻に映っている方」を返す。切り上げると、
    フレーム送りで 1 枚飛ぶ。
    """
    if ms < 0:
        return 0
    target_ticks = ms / 1000.0 * info["timescale"]
    # 浮動小数の往復で 1 目盛分を割り込むことがあるので、ごく小さな余裕を足す。
    target_ticks += 1e-6

    ticks = 0
    index = 0
    for count, delta in info["tick_table"]:
        span = count * delta
        if ticks + span > target_ticks:
            return index + int((target_ticks - ticks) // delta)
        ticks += span
        index += count
    return info["frame_count"] - 1


def spec_for_fingerprint(path: str | Path) -> dict:
    """指紋の材料を動画から作る。`fingerprint_file(path, **spec)` へそのまま渡せる。

    寸法は**回転を当てたあと**の見た目のサイズを使う。符号化サイズを使うと、
    同じ映像が向きの扱いだけで別の指紋になりうる。
    """
    info = probe_video(path)
    return {
        "duration_ms": int(round(info["duration_ms"])),
        "width": info["display_width"],
        "height": info["display_height"],
        "fps": info["fps"],
    }


def _find_moov(handle, size: int) -> tuple[int, int] | None:
    """最上位の箱を走査して moov の範囲を返す。

    iPhone の録画は mdat のあとに moov を置くので、先頭だけ読んでも見つからない。
    """
    offset = 0
    while offset + 8 <= size:
        handle.seek(offset)
        head = handle.read(8)
        if len(head) < 8:
            return None
        box_size = struct.unpack(">I", head[:4])[0]
        kind = head[4:8]
        body = offset + 8
        if box_size == 1:
            extended = handle.read(8)
            if len(extended) < 8:
                return None
            box_size = struct.unpack(">Q", extended)[0]
            body = offset + 16
        elif box_size == 0:
            box_size = size - offset
        if box_size < 8 or offset + box_size > size:
            return None
        if kind == b"moov":
            return body, offset + box_size
        offset += box_size
    return None


def _boxes(buffer: bytes, start: int, stop: int):
    """(種類, 本体の開始, 箱の終わり) を順に返す。"""
    offset = start
    while offset + 8 <= stop:
        box_size = struct.unpack(">I", buffer[offset:offset + 4])[0]
        kind = buffer[offset + 4:offset + 8]
        body = offset + 8
        if box_size == 1:
            box_size = struct.unpack(">Q", buffer[offset + 8:offset + 16])[0]
            body = offset + 16
        elif box_size == 0:
            box_size = stop - offset
        if box_size < 8 or offset + box_size > stop:
            return
        yield kind, body, offset + box_size
        offset += box_size


def _video_track(moov: bytes):
    """映像トラックだけを選んで (timescale, duration, 目盛表, 幅, 高さ, 回転) を返す。

    **`hdlr` が `vide` であることで選ぶ。** 最初に見つけた `mdhd` を採ると、
    音声トラックが先にあるファイルで音声の目盛（44100 など）を映像のものとして
    返してしまう。アプリ内の動画は映像トラックだけなので表面化していなかったが、
    アプリ外で撮った動画は音声を持つ。
    """
    for kind, body, stop in _boxes(moov, 0, len(moov)):
        if kind != b"trak":
            continue
        found = _read_track(moov, body, stop)
        if found is not None:
            return found
    return None


def _read_track(buffer: bytes, start: int, stop: int):
    dimensions = None
    for kind, body, box_stop in _boxes(buffer, start, stop):
        if kind == b"tkhd":
            dimensions = _read_tkhd(buffer, body)
        elif kind == b"mdia":
            media = _read_mdia(buffer, body, box_stop)
            if media is None:
                return None
            timescale, duration, table = media
            width, height, rotation = dimensions or (0, 0, 0)
            return timescale, duration, table, width, height, rotation
    return None


def _read_mdia(buffer: bytes, start: int, stop: int):
    timescale = duration = None
    table = None
    is_video = False
    for kind, body, box_stop in _boxes(buffer, start, stop):
        if kind == b"mdhd":
            timescale, duration = _read_mdhd(buffer, body)
        elif kind == b"hdlr":
            is_video = buffer[body + 8:body + 12] == b"vide"
        elif kind == b"minf":
            table = _find_stts(buffer, body, box_stop)
    if not is_video or timescale is None or not table:
        return None
    return timescale, duration, table


def _read_mdhd(buffer: bytes, body: int) -> tuple[int, int]:
    """version 0（32bit）と version 1（64bit）の両方を読む。

    任意の .mp4 は version 1 を使うことがある。version 0 の読み方だけだと、
    まったく違う位置を timescale として読んでしまう。
    """
    version = buffer[body]
    if version == 1:
        # version(1) flags(3) created(8) modified(8) timescale(4) duration(8)
        offset = body + 4 + 16
        timescale = struct.unpack(">I", buffer[offset:offset + 4])[0]
        duration = struct.unpack(">Q", buffer[offset + 4:offset + 12])[0]
    else:
        # version(1) flags(3) created(4) modified(4) timescale(4) duration(4)
        offset = body + 4 + 8
        timescale, duration = struct.unpack(">II", buffer[offset:offset + 8])
    return timescale, duration


def _find_stts(buffer: bytes, start: int, stop: int):
    for kind, body, box_stop in _boxes(buffer, start, stop):
        if kind == b"stts":
            return _read_stts(buffer, body)
        if kind in CONTAINERS:
            found = _find_stts(buffer, body, box_stop)
            if found:
                return found
    return None


def _read_stts(buffer: bytes, body: int) -> list[tuple[int, int]]:
    """フレーム間隔の表をランレングスのまま返す。

    展開しない。13738 枚の動画でも 2 区間で足りる。
    """
    count = struct.unpack(">I", buffer[body + 4:body + 8])[0]
    table = []
    for index in range(count):
        offset = body + 8 + index * 8
        samples, delta = struct.unpack(">II", buffer[offset:offset + 8])
        if samples > 0:
            table.append((samples, delta))
    return table


def _read_tkhd(buffer: bytes, body: int) -> tuple[int, int, int]:
    """表示サイズと回転角。行列から向きを判定する。

    metadata の解像度だけを見ていると、回転情報の書き忘れ（＝映像が寝ている）に
    気づけない（make-viewer.py の read_mov_rotation と同じ判定）。
    """
    version = buffer[body]
    # version(1) flags(3) created modified track_id reserved duration
    offset = body + 4 + (8 + 8 + 4 + 4 + 8 if version == 1 else 4 + 4 + 4 + 4 + 4)
    # reserved(8) layer(2) alternate_group(2) volume(2) reserved(2)
    offset += 8 + 2 + 2 + 2 + 2
    matrix = struct.unpack(">9i", buffer[offset:offset + 36])
    width, height = struct.unpack(">II", buffer[offset + 36:offset + 44])
    a, b, c, d = (matrix[0] / 65536, matrix[1] / 65536,
                  matrix[3] / 65536, matrix[4] / 65536)
    rounded = (round(a), round(b), round(c), round(d))
    rotation = {(0, 1, -1, 0): 90, (-1, 0, 0, -1): 180, (0, -1, 1, 0): 270}.get(rounded, 0)
    return width >> 16, height >> 16, rotation
