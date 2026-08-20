"""Extract ELF slide embeddings from tile-level foundation-model h5 files.

Uses interpolate → LayerNorm → 8-head ABMIL (the trained inference graph).
Does not load the training MoCo modules.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import load_encoder, preprocess_patch_features

VALID_FMS = ("uni", "gigapath", "h0", "virchow2", "conch_v1_5")


def save_hdf5(output_path, asset_dict, attr_dict=None, mode="w", chunk_size=32):
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with h5py.File(output_path, mode) as file:
        for key, val in asset_dict.items():
            val = np.asarray(val)
            data_shape = val.shape
            if key not in file:
                chunk_shape = (min(chunk_size, data_shape[0]),) + data_shape[1:]
                dset = file.create_dataset(
                    key, shape=data_shape, maxshape=(None,) + data_shape[1:],
                    chunks=chunk_shape, dtype=val.dtype,
                )
                dset[:] = val
                if attr_dict is not None and key in attr_dict:
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val
    return output_path


def load_patch_features(h5_path: Path, foundation_model: str | None) -> np.ndarray:
    with h5py.File(h5_path, "r") as handle:
        if "features" not in handle:
            raise KeyError(f"{h5_path} has no dataset 'features'")
        features = handle["features"][:]
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    if features.ndim != 2:
        raise ValueError(f"{h5_path}: expected [N, C] features, got {features.shape}")
    tensor = preprocess_patch_features(torch.from_numpy(features), foundation_model)
    return tensor.numpy()


@torch.inference_mode()
def encode_slide(model, features: np.ndarray, device: torch.device):
    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
    if x.ndim != 2:
        raise ValueError(f"expected [N, C], got {tuple(x.shape)}")
    x = x.unsqueeze(0)
    lens = torch.tensor([x.shape[-1]], device=device)
    features_dim, features_768, attention = model(x, lens=lens)
    return (
        features_dim.detach().float().cpu().numpy(),
        features_768.detach().float().cpu().numpy(),
        attention.detach().float().cpu().numpy(),
    )


elf_dir = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser("ELF Feature Extraction")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(elf_dir / "checkpoints" / "elf_slide_encoder.pth"),
        help="inference weights (elf_slide_encoder.pth)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="directory of tile-feature h5 files (dataset 'features')",
    )
    parser.add_argument(
        "--feature-root",
        type=str,
        default=None,
        help="parent of <fm>/<dataset>/h5_files; used with --dataset and --feature-models",
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--feature-models",
        nargs="+",
        default=None,
        help="one or more of: " + " ".join(VALID_FMS),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="output directory for a single foundation model",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="writes <output-root>/<fm>_elf when extracting several models",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_jobs(args):
    fms = list(args.feature_models or [])
    jobs = []

    if args.input_dir:
        if len(fms) > 1:
            raise ValueError("--input-dir extracts one foundation model; pass a single --feature-models")
        fm = fms[0] if fms else None
        if fm is None and "virchow2" in Path(args.input_dir).as_posix().lower():
            fm = "virchow2"
        out = args.output_path or args.output_root
        if not out:
            raise ValueError("specify --output-path")
        jobs.append((fm, Path(args.input_dir), Path(out)))
        return jobs

    if not args.feature_root or not args.dataset:
        raise ValueError("specify --input-dir, or --feature-root and --dataset")
    if not fms:
        raise ValueError("specify --feature-models")

    root = Path(args.feature_root)
    for fm in fms:
        if fm not in VALID_FMS:
            raise ValueError(f"unknown foundation model {fm!r}; expected one of {VALID_FMS}")
        input_dir = root / fm / args.dataset / "h5_files"
        if args.output_path and len(fms) == 1:
            output_dir = Path(args.output_path)
        else:
            output_root = Path(args.output_root or args.output_path or ".")
            output_dir = output_root / f"{fm}_elf"
        jobs.append((fm, input_dir, output_dir))
    return jobs


def extract_one(model, device, fm, input_dir: Path, output_dir: Path, overwrite: bool):
    if not input_dir.is_dir():
        raise FileNotFoundError(f"missing input dir: {input_dir}")
    slides = sorted(p for p in input_dir.iterdir() if p.suffix == ".h5")
    if not slides:
        raise FileNotFoundError(f"no .h5 files in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    print(f"fm={fm}  n={len(slides)}  {input_dir} -> {output_dir}")
    for h5_path in tqdm(slides, desc=f"ELF {fm or 'unknown'}"):
        out_path = output_dir / h5_path.name
        if out_path.exists() and not overwrite:
            print(f"{h5_path.name} already exists")
            continue
        patches = load_patch_features(h5_path, fm)
        features_dim, features_768, attention = encode_slide(model, patches, device)
        save_hdf5(
            out_path,
            {
                "features_dim": features_dim,
                "features": features_768,
                "attention_weights": attention,
            },
        )


def main():
    args = parse_args()

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    jobs = resolve_jobs(args)
    model = load_encoder(ckpt, device=device)
    print("Model loaded successfully")
    print(f"device={device}  checkpoint={ckpt}")
    for fm, input_dir, output_dir in jobs:
        extract_one(model, device, fm, input_dir, output_dir, args.overwrite)


if __name__ == "__main__":
    main()
