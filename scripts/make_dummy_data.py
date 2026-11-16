"""動作確認用のダミーデータを生成する

tests/dummy/
  videos/vid_XX.mp4   数秒の合成動画（正常: 灰色背景を白い四角が移動 / 異常: 途中で赤い大きな物体が出現）
  label/vid_XX.npy    異常動画のみ。フレーム単位 0/1
  splits/splits_N.txt 5 分割
"""
import argparse
import pathlib

import cv2
import numpy as np


def make_video(path, n_frames, fps, anomaly_range=None, size=(224, 160), seed=0):
    rng = np.random.default_rng(seed)
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    x, y, vx, vy = 20, 40, 3, 2
    for t in range(n_frames):
        img = np.full((h, w, 3), 110, np.uint8)
        img += rng.integers(0, 8, img.shape, dtype=np.uint8)
        x, y = x + vx, y + vy
        if not 0 < x < w - 30:
            vx = -vx
        if not 0 < y < h - 30:
            vy = -vy
        cv2.rectangle(img, (x, y), (x + 30, y + 30), (255, 255, 255), -1)
        if anomaly_range and anomaly_range[0] <= t < anomaly_range[1]:
            cv2.circle(img, (w // 2, h // 2), 50, (0, 0, 255), -1)        # BGR の赤
            cv2.putText(img, "FIRE", (w // 2 - 35, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        vw.write(img)
    vw.release()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tests/dummy")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--normal-per-split", type=int, default=2)
    p.add_argument("--anomaly-per-split", type=int, default=2)
    p.add_argument("--frames", type=int, default=72)
    p.add_argument("--fps", type=int, default=24)
    a = p.parse_args()

    out = pathlib.Path(a.out)
    for d in ("videos", "label", "splits"):
        (out / d).mkdir(parents=True, exist_ok=True)
    vid = 0
    for s in range(a.n_splits):
        names = []
        for k in range(a.normal_per_split + a.anomaly_per_split):
            name = f"vid_{vid:02d}"
            is_anom = k >= a.normal_per_split
            n = a.frames + 6 * (vid % 3)          # 長さを少しずつ変える
            rng = (n // 3, 2 * n // 3) if is_anom else None
            make_video(out / "videos" / f"{name}.mp4", n, a.fps, rng, seed=vid)
            if is_anom:
                lab = np.zeros(n, np.int64)
                lab[rng[0]:rng[1]] = 1
                np.save(out / "label" / f"{name}.npy", lab)
            names.append(f"videos/{name}.mp4")
            vid += 1
        (out / "splits" / f"splits_{s}.txt").write_text("\n".join(names) + "\n")
    print(f"生成: {vid} 本 → {out}")


if __name__ == "__main__":
    main()
