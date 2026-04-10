"""
make_splits.py
-------------
从原始 Data/四类文件夹（Kaggle imagesoasis）生成 splits/train|val|test 目录结构。
特点：
- 固定随机种子，可复现
- 只拷贝/硬链接/符号链接（三选一），默认 copy（Windows 兼容）
- 自动跳过非图片文件
用法（Windows PowerShell）：
  python make_splits.py --data_dir "D:\Code\FYP\Data" --out_dir "D:\Code\FYP\splits" --seed 42 --train 0.8 --val 0.1
"""

import os
import random
import shutil
import argparse
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_file(src: Path, dst: Path, mode: str = "copy"):
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="原始 Data 目录（含4个类别子文件夹）")
    ap.add_argument("--out_dir", type=str, required=True, help="输出 splits 目录")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--mode", type=str, default="copy", choices=["copy", "hardlink", "symlink"])
    ap.add_argument("--classes", type=str, default="", help="可选：用逗号指定类别子文件夹名（默认自动扫描）")
    args = ap.parse_args()

    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    if args.train <= 0 or args.val < 0 or args.train + args.val >= 1.0:
        raise ValueError("请保证 train>0, val>=0, 且 train+val < 1")

    # Determine classes
    if args.classes.strip():
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])

    print("Classes:", classes)

    # Create split dirs
    for split in ["train", "val", "test"]:
        for c in classes:
            ensure_dir(out_dir / split / c)

    total = 0
    for c in classes:
        class_dir = data_dir / c
        images = [p for p in class_dir.iterdir() if p.is_file() and is_image(p)]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        n_test = n - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for p in train_imgs:
            copy_file(p, out_dir / "train" / c / p.name, mode=args.mode)
        for p in val_imgs:
            copy_file(p, out_dir / "val" / c / p.name, mode=args.mode)
        for p in test_imgs:
            copy_file(p, out_dir / "test" / c / p.name, mode=args.mode)

        total += n
        print(f"{c}: train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)} (total={n})")

    print("Done. Total images:", total)
    print("Output:", out_dir)

if __name__ == "__main__":
    main()
