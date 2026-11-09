#!/usr/bin/env python3
"""Turn an image into a TV-ready animated video for Q's TV layer (laptop only: needs Pillow and ffmpeg).

    python3 scripts/build_tv_media.py IMAGE [--out tv_media/welcome.mp4] [--secs 12] [--bg white]

The image (any format Pillow reads, AVIF included) is centred on a 16:9 canvas in its own background colour, then
encoded as H.264 1920x1080 30 fps with a fade in from the background, a slow zoom, and a fade out at the end.
This is the format the office Samsung plays reliably over DLNA (the photo viewer is the flaky path).
Also writes a still 1920x1080 PNG next to the video (same name, .png) for --image use.
"""
import argparse
import os
import subprocess
import sys
import tempfile

from PIL import Image


def corner_colour(im):
    """The image's background: the most common colour among its four corners."""
    w, h = im.size
    px = [im.getpixel(p) for p in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]
    return max(set(px), key=px.count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tv_media", "welcome.mp4"))
    ap.add_argument("--secs", type=float, default=12.0, help="video length (default 12 s)")
    ap.add_argument("--zoom", type=float, default=0.08, help="how much it grows over the clip (default 0.08 = 8%%)")
    ap.add_argument("--fill", type=float, default=0.75, help="share of the screen width the image takes (default 0.75)")
    ap.add_argument("--bg", default=None, help="canvas colour, e.g. white or #102030 (default: the image's corner colour)")
    a = ap.parse_args()

    im = Image.open(a.image).convert("RGB")
    bg = a.bg or corner_colour(im)
    W, H = 3840, 2160                                   # build at 4K so the zoom has no visible stepping at 1080p
    scale = min(W * a.fill / im.width, H * 0.85 / im.height)
    art = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), bg)
    canvas.paste(art, ((W - art.width) // 2, (H - art.height) // 2))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    still = os.path.splitext(a.out)[0] + ".png"
    canvas.resize((1920, 1080), Image.LANCZOS).save(still)

    fps, n = 30, int(round(a.secs * 30))
    hexbg = bg if isinstance(bg, str) else "0x%02x%02x%02x" % bg
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "canvas.png")
        canvas.save(src)
        vf = ("zoompan=z='1+%f*on/%d':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=%d:s=1920x1080:fps=%d,"
              "fade=t=in:st=0:d=1.2:color=%s,fade=t=out:st=%.2f:d=1:color=%s,format=yuv420p"
              % (a.zoom, n, n, fps, hexbg, a.secs - 1.0, hexbg))
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", str(fps), "-i", src, "-vf", vf,
               "-frames:v", str(n), "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high",
               "-level", "4.1", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
        if subprocess.call(cmd) != 0:
            sys.exit("ffmpeg failed")
    print("video: %s (%d KB, %.0f s)\nstill: %s" % (a.out, os.path.getsize(a.out) // 1024, a.secs, still))


if __name__ == "__main__":
    main()
