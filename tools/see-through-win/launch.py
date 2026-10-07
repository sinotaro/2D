"""see-through の推論スクリプトを Radeon 向けの調整を入れてから実行する。

    python launch.py <see-through のスクリプト> [スクリプトの引数...]

Windows 版 ROCm の PyTorch では、RX 6000 シリーズ (gfx103x) 向けの省メモリな
attention カーネルが入っておらず、高解像度で attention 行列を丸ごと作って VRAM が溢れる。
そこで scaled_dot_product_attention を query 方向に分割して計算するよう差し替える
（結果は同じで、一度に使うメモリだけが減る）。
チャンクの大きさは環境変数 SEETHROUGH_SDPA_CHUNK_MB（既定 512、0 で無効）で変えられる。
"""
import os
import runpy
import sys

import torch
import torch.nn.functional as F

CHUNK_BYTES = int(os.environ.get("SEETHROUGH_SDPA_CHUNK_MB", "512")) * 2**20
_sdpa = F.scaled_dot_product_attention


def chunked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
    lq, lk = query.shape[-2], key.shape[-2]
    rows = query.numel() // (lq * query.shape[-1])  # batch * heads
    # attention スコアと softmax 用に float32 で 2 枚分を見積もる
    chunk = CHUNK_BYTES // max(1, rows * lk * 8)
    if is_causal or not query.is_cuda or chunk >= lq:
        return _sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                     is_causal=is_causal, scale=scale, **kwargs)
    chunk = max(1, chunk)
    out = []
    for i in range(0, lq, chunk):
        mask = attn_mask
        if mask is not None and mask.dim() >= 2 and mask.shape[-2] == lq:
            mask = mask[..., i:i + chunk, :]
        out.append(_sdpa(query[..., i:i + chunk, :], key, value, attn_mask=mask,
                         dropout_p=dropout_p, scale=scale, **kwargs))
    return torch.cat(out, dim=-2)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    script = os.path.abspath(sys.argv[1])
    if CHUNK_BYTES > 0:
        F.scaled_dot_product_attention = chunked_sdpa
    # 通常の `python script.py` と同じく、スクリプトのフォルダから import できるようにする
    sys.path.insert(0, os.path.dirname(script))
    sys.argv = [script] + sys.argv[2:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
