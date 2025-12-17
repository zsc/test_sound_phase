#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import html as html_lib
import io
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import stft


def load_audio_ffmpeg(path: str, sr: int = 22050, mono: bool = True, max_seconds: float | None = None):
    """
    用 ffmpeg 解码任意音频到 float32 PCM。
    - sr: 输出采样率（会重采样到该值）
    - mono: True 时输出单声道
    - max_seconds: 只取前 N 秒（可选，防止太大）
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(path),
        "-vn",
    ]

    if mono:
        cmd += ["-ac", "1"]

    cmd += ["-ar", str(sr)]

    if max_seconds is not None:
        cmd += ["-t", str(float(max_seconds))]

    # 输出为 raw float32 little-endian
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-"]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg decode failed (code={proc.returncode}):\n{err}")

    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if y.size == 0:
        raise ValueError("Decoded audio is empty (maybe unsupported file or decode failed).")

    return y, sr


def audio_to_3ch_image(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int | None = None,
    top_db: float = 80.0,
    flip_freq: bool = True,
) -> np.ndarray:
    """
    把音频转成 3 通道图像（H, W, 3），dtype=uint8。
    - R: log-magnitude spectrogram（归一化到 0..1）
    - G: (cos(phase)+1)/2
    - B: (sin(phase)+1)/2
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size < 1:
        raise ValueError("Empty audio array.")

    hop_length = hop_length or (n_fft // 4)
    if not (0 < hop_length < n_fft):
        raise ValueError("hop_length must be in (0, n_fft).")

    # 太短补零，避免 stft 自动缩小 nperseg 导致尺寸不一致
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size), mode="constant")

    noverlap = n_fft - hop_length

    # Z: (freq_bins, frames) complex
    _, _, Z = stft(
        y,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )

    mag = np.abs(Z)
    phase = np.angle(Z)

    # ---- R: log-magnitude -> 归一化到 0..1 ----
    eps = np.finfo(np.float32).eps
    mag_db = 20.0 * np.log10(np.maximum(mag, eps))
    mag_db -= mag_db.max()  # 最大值归一到 0 dB
    mag_db = np.maximum(mag_db, -float(top_db))  # 截断动态范围到 [-top_db, 0]
    mag_norm = (mag_db + float(top_db)) / float(top_db)  # 0..1

    # ---- G/B: cos/sin phase 映射到 0..1 ----
    cos_phase = (np.cos(phase) + 1.0) / 2.0
    sin_phase = (np.sin(phase) + 1.0) / 2.0

    img = np.stack([mag_norm, cos_phase, sin_phase], axis=-1)  # (H, W, 3)
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # 视觉上更常见：低频在底部
    if flip_freq:
        img_u8 = np.flipud(img_u8)

    return img_u8


def write_html_with_embedded_png(img_u8: np.ndarray, out_html: str, title: str, subtitle: str = ""):
    """
    把 RGB uint8 图像编码为 PNG(base64) 并写入自包含 HTML。
    """
    pil_img = Image.fromarray(img_u8)  # 自动识别为 RGB
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    subtitle_html = f"<p><small>{html_lib.escape(subtitle)}</small></p>" if subtitle else ""

    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>{html_lib.escape(title)}</title>
<style>
  body {{
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, "Noto Sans", "PingFang SC",
                 "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    padding: 16px;
  }}
  img {{
    max-width: 100%;
    height: auto;
    border: 1px solid #ddd;
    image-rendering: pixelated;
  }}
  code {{
    background: #f6f8fa;
    padding: 2px 4px;
    border-radius: 4px;
  }}
  small {{ color: #555; }}
</style>
</head>
<body>
  <h2>{html_lib.escape(title)}</h2>
  {subtitle_html}
  <p>
    Channels:
    <code>R = log-magnitude spectrogram</code>,
    <code>G = (cos(phase)+1)/2</code>,
    <code>B = (sin(phase)+1)/2</code>
  </p>
  <img alt="spectrogram+phase" src="data:image/png;base64,{b64}" />
</body>
</html>
"""
    Path(out_html).write_text(doc, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_audio", help="输入音频文件路径（wav/mp3/...）")
    ap.add_argument("-o", "--out_html", default="audio_3ch.html", help="输出 HTML 路径")
    ap.add_argument("--sr", type=int, default=22050, help="解码/重采样到的采样率")
    ap.add_argument("--n_fft", type=int, default=1024, help="STFT n_fft / window size")
    ap.add_argument("--hop", type=int, default=None, help="hop_length（默认 n_fft//4）")
    ap.add_argument("--top_db", type=float, default=80.0, help="log-magnitude 动态范围截断（dB）")
    ap.add_argument("--max_seconds", type=float, default=None, help="只处理前 N 秒（可选）")
    ap.add_argument("--max_width", type=int, default=None, help="可选：限制输出图像最大宽度（太宽就缩放）")
    args = ap.parse_args()

    y, sr = load_audio_ffmpeg(args.input_audio, sr=args.sr, mono=True, max_seconds=args.max_seconds)

    img = audio_to_3ch_image(
        y=y,
        sr=sr,
        n_fft=args.n_fft,
        hop_length=args.hop,
        top_db=args.top_db,
        flip_freq=True,
    )

    # 可选：如果时间轴太长，缩放到 max_width（保持高度不变）
    if args.max_width is not None and img.shape[1] > args.max_width:
        pil = Image.fromarray(img)
        pil = pil.resize((int(args.max_width), int(img.shape[0])), resample=Image.BILINEAR)
        img = np.array(pil)

    hop_used = args.hop if args.hop is not None else (args.n_fft // 4)
    subtitle = f"input={args.input_audio} | sr={sr} | n_fft={args.n_fft} | hop={hop_used} | image(H,W)={img.shape[0]},{img.shape[1]}"

    write_html_with_embedded_png(
        img_u8=img,
        out_html=args.out_html,
        title="Audio -> 3-channel (spectrogram + cos/sin phase)",
        subtitle=subtitle,
    )

    print(f"OK: wrote {args.out_html}")


if __name__ == "__main__":
    main()

