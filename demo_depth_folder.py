import argparse
import os
from pathlib import Path

import imageio
import numpy as np

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.constants import DEFAULT_MODEL
from depth_anything_3.utils.visualize import visualize_depth


def _list_images(input_dir: str, exts: str) -> list[str]:
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"input_dir must be an existing directory, got: {input_dir}")

    extensions = [e.strip().lower() for e in exts.split(",") if e.strip()]
    extensions = [e if e.startswith(".") else f".{e}" for e in extensions]

    files: list[Path] = []
    for ext in extensions:
        files.extend(input_path.glob(f"*{ext}"))
        files.extend(input_path.glob(f"*{ext.upper()}"))

    files = sorted(set(files))
    if not files:
        raise ValueError(f"No images found in {input_dir} with extensions={extensions}")
    return [str(p) for p in files]


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _to_uint8_rgb(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    x = np.clip(x, 0, 255)
    return x.astype(np.uint8)


def run(
    input_dir: str,
    output_dir: str,
    model_dir: str,
    device: str,
    process_res: int,
    process_res_method: str,
    batch_size: int,
    exts: str,
    output_ext: str,
    concat_input: bool,
) -> None:
    image_paths = _list_images(input_dir, exts)
    _ensure_dir(output_dir)

    model = DepthAnything3.from_pretrained(model_dir).to(device)

    output_ext = output_ext.lower().lstrip(".")
    if output_ext not in {"png", "jpg", "jpeg"}:
        raise ValueError("output_ext must be one of: png, jpg, jpeg")

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        pred = model.inference(
            image=batch_paths,
            export_dir=None,
            export_format="mini_npz",
            process_res=process_res,
            process_res_method=process_res_method,
        )

        for i, img_path in enumerate(batch_paths):
            stem = Path(img_path).stem
            depth_vis = visualize_depth(pred.depth[i])
            depth_vis = _to_uint8_rgb(depth_vis)

            if concat_input:
                if pred.processed_images is None:
                    raise RuntimeError("processed_images missing in prediction")
                image_vis = _to_uint8_rgb(pred.processed_images[i])
                depth_vis = np.concatenate([image_vis, depth_vis], axis=1)

            out_path = Path(output_dir) / f"{stem}_depth.{output_ext}"
            if output_ext in {"jpg", "jpeg"}:
                imageio.imwrite(out_path, depth_vis, quality=95)
            else:
                imageio.imwrite(out_path, depth_vis)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Input folder containing images")
    parser.add_argument("--output_dir", required=True, help="Output folder")
    parser.add_argument(
        "--model_dir",
        default=DEFAULT_MODEL,
        help="HuggingFace model id or local model directory",
    )
    parser.add_argument("--device", default="cuda", help="Device, e.g. cuda or cpu")
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--process_res_method", default="upper_bound_resize")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--exts", default="png,jpg,jpeg", help="Comma-separated extensions")
    parser.add_argument("--output_ext", default="png", help="png/jpg/jpeg")
    parser.add_argument("--concat_input", action="store_true")

    args = parser.parse_args()

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        device=args.device,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        batch_size=args.batch_size,
        exts=args.exts,
        output_ext=args.output_ext,
        concat_input=args.concat_input,
    )


if __name__ == "__main__":
    main()
