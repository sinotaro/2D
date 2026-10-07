#!/usr/bin/env python3
"""PSD のレイヤーを PNG に書き出し、model.json（レイヤー配置＋目・口の位置）を生成する。

使い方:
    pip install psd-tools numpy
    python3 tools/extract_psd.py path/to/model.psd [assets/model]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from psd_tools import PSDImage


def slug(name):
    return name.strip().replace(" ", "_")


def alpha_bbox(alpha, ox, oy, thr=40):
    ys, xs = np.nonzero(alpha > thr)
    if len(xs) == 0:
        return None
    return [int(xs.min() + ox), int(ys.min() + oy), int(xs.max() + ox + 1), int(ys.max() + oy + 1)]


def split_lr(layer, cx):
    """左右に分かれたパーツ（目・眉）を画面中心 cx で分割して bbox を返す。"""
    a = np.array(layer.topil().convert("RGBA"))[:, :, 3]
    mid = max(0, min(a.shape[1], int(cx - layer.left)))
    return {
        "L": alpha_bbox(a[:, :mid], layer.left, layer.top),
        "R": alpha_bbox(a[:, mid:], layer.left + mid, layer.top),
    }


def fill_hair_gaps(out, model, close=11, fringe=30):
    """髪の内側にある小さなすき間（背景が透けて見える穴）を後ろ髪の色で埋める。
    前髪・顔の裏側も後ろ髪で埋めておく。外側の毛先の間（輪郭のギザギザ）は埋めない。"""
    by = {L["name"]: L for L in model["layers"]}
    back = by.get("back hair")
    if not back:
        return
    W, H = model["width"], model["height"]
    mask = Image.new("L", (W, H), 0)
    for name in ("back hair", "front hair", "face", "ears"):
        L = by.get(name)
        if L:
            a = Image.open(out / L["file"]).getchannel("A")
            mask.paste(Image.fromarray(np.maximum(
                np.array(mask.crop((L["x"], L["y"], L["x"] + L["w"], L["y"] + L["h"]))), np.array(a))),
                (L["x"], L["y"]))
    binary = mask.point(lambda v: 255 if v > 128 else 0)
    closed = binary.filter(ImageFilter.MaxFilter(close)).filter(ImageFilter.MinFilter(close))
    # 輪郭から fringe px より内側だけを対象にする
    core = binary.filter(ImageFilter.MaxFilter(41)).filter(ImageFilter.MinFilter(41))
    for _ in range(fringe // 10):
        core = core.filter(ImageFilter.MinFilter(21))
    # 頭の内側では、後ろ髪だけで穴がないようにする（向きを変えて前髪がずれても透けない）
    holes = (np.array(closed) > 0) & (np.array(core) > 0)
    # 穴のふちを少し広げてなじませる
    holes = np.array(Image.fromarray(holes.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))) > 0
    if not holes.any():
        return
    img = Image.open(out / back["file"]).convert("RGBA")
    arr = np.array(img)
    opaque = arr[:, :, 3] > 200
    color = np.median(arr[opaque][:, :3], axis=0).astype(np.uint8)
    sub = holes[back["y"]:back["y"] + back["h"], back["x"]:back["x"] + back["w"]]
    fill = sub & (arr[:, :, 3] < 255)
    arr[fill, :3] = np.where(arr[fill, 3:4] > 0, arr[fill, :3], color)
    arr[fill, 3] = 255
    Image.fromarray(arr).save(out / back["file"], optimize=True)
    print(f"filled {int(fill.sum())} px of hair gaps")


def main():
    psd_path = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "assets/model")
    out.mkdir(parents=True, exist_ok=True)

    psd = PSDImage.open(psd_path)
    layers, by_name = [], {}
    for layer in psd.descendants():
        if layer.is_group() or not layer.visible or layer.width == 0:
            continue
        file = slug(layer.name) + ".png"
        layer.topil().convert("RGBA").save(out / file, optimize=True)
        layers.append({
            "name": layer.name,
            "file": file,
            "x": layer.left, "y": layer.top,
            "w": layer.width, "h": layer.height,
            "opacity": layer.opacity / 255,
        })
        by_name[layer.name] = layer

    model = {"width": psd.width, "height": psd.height, "layers": layers}

    # 顔の中心（左右分割の基準）
    face = by_name.get("face")
    cx = (face.left + face.right) / 2 if face else psd.width / 2
    model["faceBox"] = list(face.bbox) if face else None

    if "eyewhite" in by_name:
        eyes = split_lr(by_name["eyewhite"], cx)
        lash = split_lr(by_name["eyelash"], cx) if "eyelash" in by_name else eyes
        model["eyes"] = {
            side: {"white": eyes[side], "lash": lash[side]} for side in ("L", "R") if eyes[side]
        }
    if "eyebrow" in by_name:
        model["brows"] = split_lr(by_name["eyebrow"], cx)
    if "mouth" in by_name:
        m = by_name["mouth"]
        model["mouth"] = list(m.bbox)

    fill_hair_gaps(out, model)
    (out / "model.json").write_text(json.dumps(model, indent=2, ensure_ascii=False))
    print(f"wrote {len(layers)} layers to {out}")


if __name__ == "__main__":
    main()
