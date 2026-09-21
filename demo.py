import argparse
import os
import sys
from pathlib import Path

import cv2
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run AerialFusion registration and fusion on paired image folders.")
    parser.add_argument("--input_1", required=True, help="Visible image folder")
    parser.add_argument("--input_2", required=True, help="Infrared image folder")
    parser.add_argument("--output_dir", default="outputs/demo", help="Directory for fused results")
    parser.add_argument("--resize", nargs=2, type=int, default=[640, 480], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--max_num_keypoints", type=int, default=2048)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu. CUDA_VISIBLE_DEVICES can also be used.")
    parser.add_argument("--limit", type=int, default=-1, help="Maximum number of pairs to process; -1 means all")
    return parser.parse_args()


def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in Path(folder).iterdir() if p.suffix.lower() in exts]

    def key(path):
        try:
            return (0, int(path.stem))
        except ValueError:
            return (1, path.name)

    return sorted(files, key=key)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))

    if args.device != "cuda":
        os.environ["AERIALFUSION_DEVICE"] = args.device

    from models.extractors.superpoint import SuperPoint
    from models.matchers.AerialFusion.AerialFusion import AerialFusion
    from models.utils_tools import load_image

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vi_files = list_images(args.input_1)
    ir_files = list_images(args.input_2)
    pairs = list(zip(vi_files, ir_files))
    if args.limit > 0:
        pairs = pairs[:args.limit]
    if not pairs:
        raise RuntimeError("No paired images found.")

    width, height = args.resize
    extractor = SuperPoint(max_num_keypoints=args.max_num_keypoints).eval().to(device)
    matcher = AerialFusion(pretrained="superpoint", width_confidence=0.99, depth_confidence=0.99).eval().to(device)

    print(f"Device: {device}")
    print(f"Pairs: {len(pairs)}")
    print(f"Output: {output_dir}")

    with torch.no_grad():
        for index, (vi_path, ir_path) in enumerate(pairs):
            _, vi_color, scales0 = load_image(vi_path, resize=[height, width])
            _, ir_color, scales1 = load_image(ir_path, resize=[height, width])

            vi_batch = vi_color.unsqueeze(0).to(device)
            ir_batch = ir_color.unsqueeze(0).to(device)
            feats0 = extractor({"image": vi_batch})
            feats1 = extractor({"image": ir_batch})

            data = {
                "img0": vi_batch,
                "img1": ir_batch,
                "image_size0": torch.tensor([[width, height]], device=device),
                "image_size1": torch.tensor([[width, height]], device=device),
                "keypoints0": feats0["keypoints"],
                "keypoints1": feats1["keypoints"],
                "descriptors0": feats0["descriptors"],
                "descriptors1": feats1["descriptors"],
                "keypoint_scores0": feats0["keypoint_scores"],
                "keypoint_scores1": feats1["keypoint_scores"],
                "modal": "multi",
            }

            pred = matcher(
                data=data,
                scales0=scales0,
                scales1=scales1,
                img_width=width,
                img_height=height,
            )
            fused = pred["return_pre"].get("Fusion")
            if isinstance(fused, torch.Tensor) or fused is None or getattr(fused, "size", 0) == 0:
                print(f"[{index}] skip: insufficient matches for {vi_path.name} / {ir_path.name}")
                continue

            out_name = f"{index:06d}.png"
            cv2.imwrite(str(output_dir / out_name), fused)
            print(f"[{index + 1}/{len(pairs)}] saved {out_name}")


if __name__ == "__main__":
    main()
