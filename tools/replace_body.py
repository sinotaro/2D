#!/usr/bin/env python3
"""1 枚絵（白背景）から首より下を切り出して、モデルの体（neck / topwear / neckwear）と差し替える。

頭（顔・髪）は今のモデルのまま使い、新しい絵の首をモデルのあごの真下につなぐ。

    python3 tools/replace_body.py 新しい絵.png --chin 680,652 --neck-center 699 --cut 738
        --chin         新しい絵のあごの先端 (x,y)
        --neck-center  あごのすぐ下での首の中心 x
        --cut          この y より上の暗い色（首にかかる髪）を消す。襟のいちばん上より少し上にする
        --scale        新しい絵 → モデルの拡大率（省略時は両目の間隔から自動）
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "assets" / "model"


def iris_centers(rgb, box):
    """青い黒目の重心を左右それぞれ返す。"""
    x0, y0, x1, y1 = box
    sub = rgb[y0:y1, x0:x1].astype(int)
    blue = (sub[:, :, 2] - sub[:, :, 0] > 50) & (sub[:, :, 2] > 120)
    ys, xs = np.nonzero(blue)
    xs, ys = xs + x0, ys + y0
    mid = (x0 + x1) / 2
    return [(xs[s].mean(), ys[s].mean()) for s in (xs < mid, xs >= mid)]


def remove_white(img, thresh=28):
    """外周からつながった白背景を透明にし、ふちの白いにじみを取り除く。"""
    rgb = img.convert("RGB")
    key = rgb.copy()
    w, h = key.size
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0), (w // 2, h - 1)]:
        if min(key.getpixel(seed)) > 240:
            ImageDraw.floodfill(key, seed, (255, 0, 255), thresh=thresh)
    k = np.array(key)
    bg = (k[:, :, 0] == 255) & (k[:, :, 1] == 0) & (k[:, :, 2] == 255)
    fg = ~bg
    arr = np.array(rgb).astype(float)
    # ふち 2px は白と混ざっているので、内側の色から不透明度を逆算する
    inner = np.array(Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5))) > 0
    band = fg & ~inner
    fgcol = np.array(Image.fromarray(np.where(inner[..., None], arr, 255).astype(np.uint8))
                     .filter(ImageFilter.MinFilter(5))).astype(float)
    alpha = np.where(fg, 1.0, 0.0)
    a_band = ((255 - arr) / np.maximum(1, 255 - fgcol)).mean(axis=2).clip(0, 1)
    alpha[band] = a_band[band]
    out = np.where(band[..., None], fgcol, arr)
    return np.dstack([out, alpha * 255]).clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--chin", required=True)
    ap.add_argument("--neck-center", type=float, required=True)
    ap.add_argument("--cut", type=int, required=True)
    ap.add_argument("--scale", type=float)
    ap.add_argument("--extend-up", type=int, default=90, help="首を上（顔の裏）に描き足す量 px")
    args = ap.parse_args()
    chin_x, chin_y = map(float, args.chin.split(","))

    model = json.loads((MODEL / "model.json").read_text())
    layers = {L["name"]: L for L in model["layers"]}
    src = Image.open(args.image)
    rgb = np.array(src.convert("RGB"))

    # 拡大率：両目の間隔をそろえる
    if args.scale:
        scale = args.scale
    else:
        L = layers["irides"]
        canvas = np.full((model["height"], model["width"], 3), 255, np.uint8)
        ir = np.array(Image.open(MODEL / L["file"]).convert("RGBA"))
        m = ir[:, :, 3] > 128
        canvas[L["y"]:L["y"] + L["h"], L["x"]:L["x"] + L["w"]][m] = ir[:, :, :3][m]
        fb = model["faceBox"]
        a1, a2 = iris_centers(canvas, (fb[0], fb[1], fb[2], fb[3]))
        b1, b2 = iris_centers(rgb, (int(chin_x - 300), int(chin_y - 300), int(chin_x + 300), int(chin_y - 80)))
        scale = np.hypot(a2[0] - a1[0], a2[1] - a1[1]) / np.hypot(b2[0] - b1[0], b2[1] - b1[1])
    print(f"scale {scale:.3f}")

    rgba = remove_white(src)
    a = rgba[:, :, 3].astype(float)
    H, W = a.shape
    yy, xx = np.mgrid[0:H, 0:W]
    # あごより上（顔）を消す
    a[yy < chin_y] = 0
    # 髪の毛先に囲まれて外周からつながっていなかった白い背景のかけらを消す（首まわりだけ）。
    # 肌は赤みがあるので、白〜明るい灰色で色みのない画素だけが対象
    c = rgba[:, :, :3].astype(int)
    whitish = (c.min(axis=2) > 225) & (c.max(axis=2) - c.min(axis=2) < 14) & (yy < args.cut + 40)
    # 透明な部分に接している白っぽい画素だけを外側から何回かはがす（首の内側の明るい肌は輪郭線で守られる）
    for _ in range(6):
        clear = a < 30
        near = np.zeros_like(clear)
        near[1:] |= clear[:-1]; near[:-1] |= clear[1:]; near[:, 1:] |= clear[:, :-1]; near[:, :-1] |= clear[:, 1:]
        peel = whitish & near & (a >= 30)
        if not peel.any():
            break
        a[peel] = 0
    rgba[:, :, 3] = a.astype(np.uint8)
    dark = rgba[:, :, :3].max(axis=2) < 110

    # 襟より上の行は「首の中心からつながった肌の区間」だけを残す。
    # 消した髪のふちの残り（灰色のにじみ）や、首から離れたゴミをここで落とす。
    top = int(chin_y)
    cx = int(args.neck_center)
    skin = (a > 200) & ~dark
    for y in range(top, args.cut):
        row_skin = skin[y]
        if not row_skin[cx]:
            continue
        lo = cx
        while lo > 0 and (row_skin[lo - 1] or (not row_skin[lo - 1] and row_skin[max(0, lo - 6):lo - 1].any())):
            lo -= 1
        hi = cx
        while hi < W - 1 and (row_skin[hi + 1] or row_skin[hi + 2:hi + 7].any()):
            hi += 1
        # 区間の内側（首の輪郭線を含む外側 3px まで）は元の絵のまま残し、外側（首にかかる髪）を消す
        lo, hi = max(0, lo - 3), min(W - 1, hi + 3)
        rgba[y, :lo, 3] = 0
        rgba[y, hi + 1:, 3] = 0
    a = rgba[:, :, 3].astype(float)
    skin = (a > 200) & ~dark

    # 首を上に描き足す（モデルの顔の裏に隠れる部分）。
    # あごのすぐ下の行を基準に、その下の首を上下反転して上へ続ける（境目で模様が途切れない）。
    # 反転する範囲は襟にかからない 45 行まで、それより上はその行をくり返す。
    ref_y = top + 14
    n_ext = args.extend_up + (ref_y - top)
    ext = np.zeros((n_ext, W, 4), np.uint8)
    for k in range(1, n_ext + 1):
        row = rgba[ref_y + min(k, 45)].copy()
        row[:, 3] = np.where(skin[ref_y + min(k, 45)] | (row[:, 3] > 200), row[:, 3], 0)
        ext[n_ext - k] = row
    body = np.concatenate([ext, rgba[ref_y:]], axis=0)
    body_top = top - args.extend_up

    # 服の下端が絵の端で切れているので、最後の行をくり返して下に伸ばす
    last = np.nonzero(body[:, :, 3].max(axis=1) > 0)[0].max()
    pad = int(200 / scale)  # 体を傾けても下端が見えないよう、キャンバスの外まで伸ばす
    # 絵の下端の数行は白とにじんでいるので、少し内側の行をくり返す
    body = np.concatenate([body[:last - 8], np.repeat(body[last - 9][None], pad, axis=0)], axis=0)

    # モデル座標へ：あごの下の首の中心 → モデルのあご中心
    fb = model["faceBox"]
    face = layers["face"]
    fa = np.array(Image.open(MODEL / face["file"]))[:, :, 3]
    ys = np.nonzero(fa.max(axis=1) > 128)[0]
    model_chin_y = face["y"] + ys.max()
    model_cx = (fb[0] + fb[2]) / 2
    img = Image.fromarray(body)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    ox = round(model_cx - args.neck_center * scale)
    oy = round(model_chin_y - (top - body_top) * scale)
    bbox = img.getbbox()
    img = img.crop(bbox)
    x, y = ox + bbox[0], oy + bbox[1]
    img.save(MODEL / "topwear.png", optimize=True)

    keep = [L for L in model["layers"] if L["name"] not in ("neck", "neckwear")]
    for L in keep:
        if L["name"] == "topwear":
            L.update({"x": int(x), "y": int(y), "w": img.width, "h": img.height})
    model["layers"] = keep
    (MODEL / "model.json").write_text(json.dumps(model, indent=2, ensure_ascii=False))
    for name in ("neck.png", "neckwear.png"):
        (MODEL / name).unlink(missing_ok=True)
    print(f"topwear at ({x},{y}) size {img.width}x{img.height}")


if __name__ == "__main__":
    main()
