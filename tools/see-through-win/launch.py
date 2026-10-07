"""see-through の推論スクリプトを Radeon 向けの調整を入れてから実行する。

    python launch.py <see-through のスクリプト> [スクリプトの引数...]

入れている調整:

1. attention の分割計算
   Windows 版 ROCm の PyTorch には RX 6000 シリーズ (gfx103x) 向けの省メモリな
   attention カーネルがなく、高解像度で attention 行列を丸ごと作って VRAM が溢れる。
   scaled_dot_product_attention を query 方向に分割して計算する（結果は同じ）。
   チャンクの大きさは環境変数 SEETHROUGH_SDPA_CHUNK_MB（既定 512、0 で無効）。

2. VAE デコードのタイル分割
   1280px の VAE デコードを一度に行うと、遅い GPU では 1 回の畳み込みが数秒かかり、
   Windows の GPU タイムアウト (TDR, 既定 2 秒) でドライバがリセットされて
   "unspecified launch failure" で落ちる。デコードだけ 512px のタイルに分けて行う。
   SEETHROUGH_VAE_TILE=0 で無効。

3. 生成結果（latent）の保存と再利用
   拡散の 30 ステップ（RX 6600 XT で 1〜2 時間）が終わった時点の latent を
   SEETHROUGH_LATENT_CACHE のフォルダに保存し、同じ画像・設定で再実行したときは
   拡散を飛ばしてデコードからやり直す。
"""
import hashlib
import os
import runpy
import sys

import torch
import torch.nn.functional as F

CHUNK_BYTES = int(os.environ.get("SEETHROUGH_SDPA_CHUNK_MB", "512")) * 2**20
VAE_TILE = os.environ.get("SEETHROUGH_VAE_TILE", "1") != "0"
LATENT_CACHE = os.environ.get("SEETHROUGH_LATENT_CACHE", "")
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


def _find_frame(*names):
    """呼び出し元をさかのぼり、指定したローカル変数を全部持つフレームを返す（パイプラインの __call__）。"""
    f = sys._getframe(2)
    while f is not None:
        if all(n in f.f_locals for n in names):
            return f
        f = f.f_back
    return None


class LatentCache:
    def __init__(self, folder):
        self.folder = folder
        self.pending_save = None  # 保存先（拡散を実行した回）
        self.cached = None        # 読み込んだ latent（拡散を飛ばした回）
        self.index = 0

    def key(self, frame):
        loc = frame.f_locals
        h = hashlib.sha1()
        page = loc.get("fullpage")
        if page is not None:
            h.update(page.tobytes())
        h.update(repr((loc.get("group_index"), loc.get("num_inference_steps"), sys.argv)).encode())
        return os.path.join(self.folder, h.hexdigest()[:16] + ".pt")

    def patch_timesteps(self):
        from diffusers.pipelines.stable_diffusion_xl import pipeline_stable_diffusion_xl_img2img as img2img
        orig = img2img.retrieve_timesteps
        cache = self

        def retrieve_timesteps(*args, **kwargs):
            timesteps, n = orig(*args, **kwargs)
            frame = _find_frame("fullpage", "group_index")
            if frame is None:
                return timesteps, n
            path = cache.key(frame)
            if os.path.exists(path):
                print(f"[launch] 保存済みの生成結果を使います（拡散をスキップ）: {path}")
                cache.cached, cache.index = torch.load(path), 0
                return timesteps[:0], 0
            cache.pending_save = path
            return timesteps, n

        img2img.retrieve_timesteps = retrieve_timesteps

    def on_decode(self, latent):
        """デコード直前に呼ばれる。保存または差し替えをした latent を返す。"""
        if self.cached is not None:
            latent = self.cached[self.index][None].to(latent)
            self.index += 1
            if self.index >= len(self.cached):
                self.cached = None
            return latent
        if self.pending_save is not None:
            frame = _find_frame("latents", "fullpage")
            if frame is not None:
                os.makedirs(self.folder, exist_ok=True)
                torch.save(frame.f_locals["latents"].detach().cpu(), self.pending_save)
                print(f"[launch] 生成結果を保存しました: {self.pending_save}")
            self.pending_save = None
        return latent


def patch_decoder(cache):
    from modules.layerdiffuse.vae import TransparentVAEDecoder
    orig = TransparentVAEDecoder.forward

    def forward(self, sd_vae, latent, *args, **kwargs):
        if cache is not None:
            latent = cache.on_decode(latent)
        if not VAE_TILE:
            return orig(self, sd_vae, latent, *args, **kwargs)
        saved = (sd_vae.use_tiling, sd_vae.tile_sample_min_size, sd_vae.tile_latent_min_size)
        sd_vae.use_tiling, sd_vae.tile_sample_min_size, sd_vae.tile_latent_min_size = True, 512, 64
        try:
            return orig(self, sd_vae, latent, *args, **kwargs)
        finally:
            sd_vae.use_tiling, sd_vae.tile_sample_min_size, sd_vae.tile_latent_min_size = saved

    TransparentVAEDecoder.forward = forward


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    script = os.path.abspath(sys.argv[1])
    if CHUNK_BYTES > 0:
        F.scaled_dot_product_attention = chunked_sdpa
    cache = LatentCache(LATENT_CACHE) if LATENT_CACHE else None
    if cache is not None:
        cache.patch_timesteps()
    patch_decoder(cache)
    # 通常の `python script.py` と同じく、スクリプトのフォルダから import できるようにする
    sys.path.insert(0, os.path.dirname(script))
    sys.argv = [script] + sys.argv[2:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
