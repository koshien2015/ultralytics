#!/usr/bin/env python3
"""動画を識別する指紋。**正準実装はこの 1 ファイルだけ。**

**1 球 1 フォルダで自己完結させる方針（設計文書 §13）になったので、
指紋の仕事は 1 つに絞られた:**

> GPU の別 PC から戻ってきた解析結果が、このフォルダのクリップに対して
> 計算されたものか。

投球ごとにクリップが違うので、ここでは指紋が実際に区別できる。
160 個のフォルダへ人が手でファイルを置く運用なので、取り違えは起こりうる。

**もうしないこと:** ファイル名や metadata.json から元動画を探して突き合わせること。
フォルダに同居しているので、探すステップ自体が無い。同一セッションの
p03 と p04 の取り違えは**動画が同じなので指紋では検出できない**——
区別するのはフォルダとクリップであって指紋ではない。

材料は `video_probe.spec_for_fingerprint()` が動画ファイル自身から作る。
metadata.json の自己申告は使わない（実データで durationMs が 5ms、
fps が 8 桁目で食い違っていた）。この計算は **2 台のマシンで別々に実行され、
文字列として突き合わせられる**ので、連結の順序・区切り・数値の表記まで
ここで固定する。再実装しないこと。

全バイトの SHA-256 は採らない。240fps の長尺 .mov は 1 セッション 100MB を超え、
41 セッションでは現実的な時間で終わらない。先頭と末尾だけを読み、
サイズ・長さ・解像度・実測 fps と合わせて複合キーにする。

この方式で区別できるもの / できないもの:
- 再エンコード      → 末尾とサイズがまず一致しない。区別できる
- トリミング        → 長さとサイズが変わる。区別できる
- 同名の別セッション → 先頭が違う。区別できる
- 同一内容のコピー   → 全て一致する。**区別しない（同じ動画なので結び付けてよい）**
- 中間フレームだけの改変 → **区別できない。** 既知の限界（test に明記してある）
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# 方式名。読む規則を変えたら必ずここも変える。古い封筒と混ざったまま
# 比較されると、不一致なのか方式違いなのか区別できなくなる。
METHOD = "capfp1"

CHUNK = 1024 * 1024        # 先頭・末尾から読む長さ
WHOLE_FILE_LIMIT = 2 * CHUNK   # これ以下なら全体を読む（先頭と末尾が重なるため）


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(head: bytes, tail: bytes, *, size_bytes: int, duration_ms: int,
            width: int, height: int, fps: float) -> str:
    """正準レシピ。ここの 1 行の順序がプロトコルそのもの。

    fps は repr で書く。丸めると 240.0 と 239.48835806346878 が同じになる
    ことはないが、239.48835806346878 と 239.48835806346875 は丸め方次第で
    同じになる。実測値はそのまま残す。
    """
    recipe = "\n".join([
        METHOD,
        _sha256_hex(head),
        _sha256_hex(tail),
        str(int(size_bytes)),
        str(int(duration_ms)),
        f"{int(width)}x{int(height)}",
        repr(float(fps)),
    ])
    return f"{METHOD}:{_sha256_hex(recipe.encode('utf-8'))}"


def _split(data: bytes) -> tuple[bytes, bytes]:
    """先頭と末尾を取り出す。重なる大きさなら全体を先頭として扱う。

    小さいファイルで先頭 1MiB と末尾 1MiB を別々に取ると、同じバイトを
    2 度数えることになる。実装ごとに扱いが割れる場所なので、
    「2MiB 以下は全体・末尾は空」と決め打つ。
    """
    if len(data) <= WHOLE_FILE_LIMIT:
        return data, b""
    return data[:CHUNK], data[-CHUNK:]


def fingerprint_bytes(data: bytes, *, duration_ms: int, width: int, height: int,
                      fps: float) -> str:
    """バイト列から指紋を作る。テストと小さいファイル用。"""
    head, tail = _split(data)
    return _digest(head, tail, size_bytes=len(data), duration_ms=duration_ms,
                   width=width, height=height, fps=fps)


def fingerprint_file(path: str | Path, *, duration_ms: int, width: int, height: int,
                     fps: float) -> str:
    """ファイルから指紋を作る。全体はメモリへ載せない。"""
    target = Path(path)
    try:
        size = target.stat().st_size
        with target.open("rb") as handle:
            if size <= WHOLE_FILE_LIMIT:
                head, tail = handle.read(), b""
            else:
                head = handle.read(CHUNK)
                handle.seek(-CHUNK, 2)
                tail = handle.read(CHUNK)
    except OSError as error:
        raise OSError(f"動画を読めないため指紋を作れない: {target}: {error}") from error
    return _digest(head, tail, size_bytes=size, duration_ms=duration_ms,
                   width=width, height=height, fps=fps)
