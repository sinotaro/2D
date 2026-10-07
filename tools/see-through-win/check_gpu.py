"""ROCm 版 PyTorch が Radeon で正しく計算できるか確かめる。

    python check_gpu.py          # 詳しいチェック
    python check_gpu.py --quick  # インストール直後の簡易チェック

GPU が使えないか、計算結果が CPU と合わなければ終了コード 1 を返す。
"""
import sys
import time

import torch
import torch.nn.functional as F


def close(a, b, tol):
    err = (a.float().cpu() - b.float().cpu()).abs().max().item()
    return err <= tol, err


def main():
    quick = "--quick" in sys.argv
    print(f"torch {torch.__version__}  (HIP {torch.version.hip})")
    if torch.version.hip is None:
        print("NG: ROCm 版ではない torch が入っています")
        return 1
    if not torch.cuda.is_available():
        print("NG: GPU が見つかりません（Adrenalin ドライバを最新にしてください）")
        return 1

    prop = torch.cuda.get_device_properties(0)
    arch = getattr(prop, "gcnArchName", "?")
    print(f"GPU: {prop.name}  arch={arch}  VRAM={prop.total_memory / 2**30:.1f} GiB")

    torch.manual_seed(0)
    checks = []

    a, b = torch.randn(512, 512), torch.randn(512, 512)
    ref = a @ b
    for dtype, tol in ((torch.float32, 1e-2), (torch.bfloat16, 2.0), (torch.float16, 0.5)):
        out = a.to("cuda", dtype) @ b.to("cuda", dtype)
        checks.append((f"matmul {dtype}", *close(out, ref, tol)))

    if not quick:
        x, w = torch.randn(1, 64, 128, 128), torch.randn(64, 64, 3, 3)
        ref = F.conv2d(x, w, padding=1)
        out = F.conv2d(x.to("cuda", torch.bfloat16), w.to("cuda", torch.bfloat16), padding=1)
        checks.append(("conv2d bf16", *close(out, ref, 2.0)))

        q, k, v = (torch.randn(1, 10, 1024, 64) for _ in range(3))
        ref = F.scaled_dot_product_attention(q, k, v)
        out = F.scaled_dot_product_attention(*(t.to("cuda", torch.bfloat16) for t in (q, k, v)))
        checks.append(("attention bf16", *close(out, ref, 0.05)))

    ok = True
    for name, passed, err in checks:
        print(f"  {'OK' if passed else 'NG'}  {name:18s} max err {err:.4g}")
        ok &= passed

    if not quick:
        x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            x @ x
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        print(f"  bf16 matmul: {10 * 2 * 4096 ** 3 / dt / 1e12:.1f} TFLOPS")

    print("OK: GPU で計算できます" if ok else "NG: GPU の計算結果がおかしいです")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
