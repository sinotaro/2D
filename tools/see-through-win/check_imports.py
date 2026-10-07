"""see-through に必要なモジュールが読み込めるか 1 つずつ確かめ、失敗したら原因を全部表示する。

    （see-through\\repo フォルダで）python check_imports.py [ログの保存先]
"""
import importlib
import io
import sys
import traceback

MODULES = [
    "torch", "torchvision", "transformers", "diffusers", "accelerate", "timm",
    "psd_tools", "cv2", "kornia",
    "transformers.models.clip",
    "diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img",
    "utils.inference_utils", "modules.layerdiffuse.layerdiff3d", "modules.marigold",
]


def main():
    log = io.StringIO()

    def out(s=""):
        print(s)
        log.write(s + "\n")

    out(f"python {sys.version}")
    ok = True
    for name in MODULES:
        try:
            m = importlib.import_module(name)
            out(f"OK  {name} {getattr(m, '__version__', '')}")
        except BaseException:
            ok = False
            out(f"NG  {name}")
            out(traceback.format_exc())

    try:
        import torch
        import torchvision
        boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float32)
        torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8]), 0.5)
        out("OK  torchvision.ops.nms")
    except BaseException:
        ok = False
        out("NG  torchvision.ops.nms（torch と torchvision の組み合わせが合っていない可能性）")
        out(traceback.format_exc())

    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as f:
            f.write(log.getvalue())
        print(f"\nログ: {sys.argv[1]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
