import argparse
import os
from pathlib import Path

import imageio
import numpy as np
from PIL import Image

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.constants import DEFAULT_MODEL
from depth_anything_3.utils.visualize import visualize_depth


def _read_image_list(txt_path: str) -> list[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    image_paths = [ln for ln in lines if ln and not ln.startswith("#")]
    return image_paths


def _list_images(image_dir: str, exts: str, *, recursive: bool) -> list[str]:
    input_path = Path(image_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"image_dir must be an existing directory, got: {image_dir}")

    extensions = [e.strip().lower() for e in exts.split(",") if e.strip()]
    extensions = [e if e.startswith(".") else f".{e}" for e in extensions]

    files: list[Path] = []
    it = input_path.rglob("*") if recursive else input_path.iterdir()
    for p in it:
        if not p.is_file():
            continue
        if p.suffix.lower() in extensions:
            files.append(p)

    files = sorted(set(files))
    if not files:
        raise ValueError(f"No images found in {image_dir} with extensions={extensions}")
    return [str(p) for p in files]


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _to_uint8_rgb(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    x = np.clip(x, 0, 255)
    return x.astype(np.uint8)


def _load_original_for_concat(img_path: str) -> np.ndarray:
    img = imageio.imread(img_path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def _resize_depth_to_match(depth: np.ndarray, *, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if depth.shape[0] == target_h and depth.shape[1] == target_w:
        return depth
    pil = Image.fromarray(depth.astype(np.float32), mode="F")
    pil = pil.resize((target_w, target_h), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.float32)


def _safe_infer(
    model: DepthAnything3,
    batch_paths: list[str],
    *,
    process_res: int,
    process_res_method: str,
) -> tuple[object | None, list[str]]:
    try:
        pred = model.inference(
            image=batch_paths,
            export_dir=None,
            export_format="mini_npz",
            process_res=process_res,
            process_res_method=process_res_method,
        )
        return pred, batch_paths
    except Exception as e:
        print(f"Warning: inference failed for a batch of {len(batch_paths)} images, will try per-image. Error: {e}")

    ok_preds: list[object] = []
    ok_paths: list[str] = []
    for p in batch_paths:
        try:
            pred = model.inference(
                image=[p],
                export_dir=None,
                export_format="mini_npz",
                process_res=process_res,
                process_res_method=process_res_method,
            )
        except Exception as e:
            print(f"Warning: skipping unreadable image: {p}. Error: {e}")
            continue
        ok_preds.append(pred)
        ok_paths.append(p)

    if len(ok_paths) == 0:
        return None, []

    pred0 = ok_preds[0]
    pred0.depth = np.stack([p.depth[0] for p in ok_preds], axis=0)
    if getattr(pred0, "processed_images", None) is not None:
        pred0.processed_images = [p.processed_images[0] for p in ok_preds]
    return pred0, ok_paths


def run(
    *,
    image_list: str | None,
    image_dir: str | None,
    out_dir: str | None,
    no_recursive: bool,
    model_dir: str,
    device: str,
    process_res: int,
    process_res_method: str,
    batch_size: int,
    exts: str,
    output_ext: str,
    concat_input: bool,
    concat_original: bool,
) -> None:
    if (image_list is None) == (image_dir is None):
        raise ValueError("Specify exactly one of --image_list or --image_dir")

    if out_dir is None:
        if image_dir is None:
            raise ValueError("--out_dir is required when using --image_list")
        out_dir_path = Path(f"{image_dir}_vggt")
    else:
        out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    image_root: Path | None = None
    if image_list is not None:
        image_paths = _read_image_list(image_list)
    else:
        image_root = Path(image_dir)
        image_paths = _list_images(image_dir, exts, recursive=(not no_recursive))
    if len(image_paths) == 0:
        src = image_list if image_list is not None else image_dir
        raise ValueError(f"No valid image paths found in {src}")

    model = DepthAnything3.from_pretrained(model_dir).to(device)

    output_ext = output_ext.lower().lstrip(".")
    if output_ext not in {"png", "jpg", "jpeg"}:
        raise ValueError("output_ext must be one of: png, jpg, jpeg")

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]

        pred, ok_paths = _safe_infer(
            model,
            batch_paths,
            process_res=process_res,
            process_res_method=process_res_method,
        )
        if pred is None or len(ok_paths) == 0:
            continue

        for i, img_path in enumerate(ok_paths):
            stem = Path(img_path).stem
            depth_vis = visualize_depth(pred.depth[i])
            depth_vis = _to_uint8_rgb(depth_vis)

            if concat_input:
                if pred.processed_images is None:
                    raise RuntimeError("processed_images missing in prediction")
                image_vis = _to_uint8_rgb(pred.processed_images[i])
                depth_vis = np.concatenate([image_vis, depth_vis], axis=1)

            if concat_original:
                try:
                    orig_vis = _load_original_for_concat(img_path)
                except Exception as e:
                    print(f"Warning: failed to read original image for concat: {img_path}. Error: {e}")
                else:
                    depth_aligned = _resize_depth_to_match(
                        pred.depth[i],
                        target_hw=(orig_vis.shape[0], orig_vis.shape[1]),
                    )
                    depth_vis_aligned = _to_uint8_rgb(visualize_depth(depth_aligned))
                    depth_vis = np.concatenate([orig_vis, depth_vis_aligned], axis=1)

            if image_root is not None:
                rel = Path(img_path).relative_to(image_root)
                stem = rel.stem
                img_out_dir = out_dir_path / rel.parent
                img_out_dir.mkdir(parents=True, exist_ok=True)
            else:
                img_out_dir = out_dir_path

            out_path = img_out_dir / f"{stem}.depth.{output_ext}"
            if output_ext in {"jpg", "jpeg"}:
                imageio.imwrite(out_path, depth_vis, quality=95)
            else:
                imageio.imwrite(out_path, depth_vis)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_list", type=str, default=None, help="Path to txt file, one image path per line")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to a directory of images")
    parser.add_argument("--input_dir", dest="image_dir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--output_dir", dest="out_dir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        default=False,
        help="Disable recursive scan when using --image_dir (recursive is enabled by default)",
    )
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
    parser.add_argument("--concat_original", action="store_true")

    args = parser.parse_args()

    run(
        image_list=args.image_list,
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        no_recursive=args.no_recursive,
        model_dir=args.model_dir,
        device=args.device,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        batch_size=args.batch_size,
        exts=args.exts,
        output_ext=args.output_ext,
        concat_input=args.concat_input,
        concat_original=args.concat_original,
    )


if __name__ == "__main__":
    main()
