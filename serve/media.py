"""Image and video content parts: fetch, decode, preprocess.

Preprocessing runs the checkpoint's own `MuseGlimmerImageProcessor` /
`MuseGlimmerVideoProcessor` rather than a re-implementation. That is the same
decision as rendering through `chat_template.jinja`: the processor is a
`TorchvisionBackend` whose `resample: 1` is torchvision's antialiased LANCZOS
(and only on torchvision >= 0.27 — below that it silently substitutes BICUBIC,
which is a difference nobody notices until captions get worse), so matching it
by hand is a standing liability with no upside here.

`input_audio` is a capability error, not a silence: this model has no audio
tower, no audio config, and no audio tensors.
"""
from __future__ import annotations

import base64
import hashlib
import io
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass

import numpy as np

MAX_BYTES = 64 * 1024 * 1024
PATCH = "<|patch|>"
VIDEO = "<|video|>"


class MediaError(ValueError):
    """A bad or unsupported media part. Surfaced as a 400."""


@dataclass
class Media:
    kind: str                      # "image" | "video"
    pixels: np.ndarray             # f64 [npatch, patch_dim]
    grid: tuple[int, int, int]     # (t, h, w) in patches

    @property
    def patches(self) -> int:
        return int(self.grid[0] * self.grid[1] * self.grid[2])

    def tokens(self, merge_unit: int) -> int:
        return self.patches // merge_unit


def fetch(url: str) -> bytes:
    """data:, file:, http(s):. Bounded, because a 4 GiB "image" should be a
    400 and not an OOM."""
    if url.startswith("data:"):
        head, _, payload = url.partition(",")
        if ";base64" in head:
            return base64.b64decode(payload)
        return urllib.parse.unquote_to_bytes(payload)
    if url.startswith(("http://", "https://", "file://")):
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise MediaError(f"media exceeds the {MAX_BYTES // (1 << 20)} MiB cap")
        return data
    raise MediaError(f"unsupported URL scheme in {url[:32]!r}")


class MediaCache:
    """Content-addressed decode cache: an agent re-sending the same screenshot
    every turn should pay for it once."""

    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self._d: dict[str, Media] = {}

    def get_or_make(self, raw: bytes, make) -> Media:
        key = hashlib.sha256(raw).hexdigest()
        hit = self._d.get(key)
        if hit is None:
            hit = make()
            if len(self._d) >= self.capacity:
                self._d.pop(next(iter(self._d)))
            self._d[key] = hit
        return hit


def decode_video(raw: bytes, *, fps: float, max_frames: int) -> np.ndarray:
    """A video container -> uint8 frames [n, h, w, 3], through the ffmpeg binary.

    Not through a Python decoder, because there is not one: `pyav`, `decord`,
    `opencv` and torchvision's video reader are all absent from the pinned
    environment, and adding a build-heavy dependency to decode a container is a
    poor trade when ffmpeg is already the thing all four of them wrap. The
    sampling (fps, frame cap) is the checkpoint processor's own, taken from
    processor_config.json, and it is applied HERE so `do_sample_frames=False`
    can keep the processor from sampling twice.
    """
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise MediaError("video input needs the ffmpeg and ffprobe binaries on PATH "
                         "(no Python video decoder is installed either); send the frames "
                         "as a list of images instead, or install ffmpeg")
    with tempfile.NamedTemporaryFile(suffix=".vid") as f:
        f.write(raw)
        f.flush()
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "csv=p=0", f.name],
            capture_output=True, text=True)
        if probe.returncode != 0 or "," not in probe.stdout:
            raise MediaError(f"ffprobe could not read this video: {probe.stderr.strip()[:200]}")
        w, h = (int(x) for x in probe.stdout.strip().split(",")[:2])
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", f.name, "-vf", f"fps={fps}",
             "-frames:v", str(max_frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True)
        if out.returncode != 0:
            raise MediaError(f"ffmpeg could not decode this video: "
                             f"{out.stderr.decode(errors='replace').strip()[:200]}")
    frame_bytes = w * h * 3
    n = len(out.stdout) // frame_bytes
    if n == 0:
        raise MediaError("video decoded to zero frames")
    # temporal_patch_size pairs frames, so an odd count would drop one
    n -= n % 2
    # np.frombuffer is read-only and the processor converts through torch,
    # which warns about that; a copy is cheaper than the warning is useful.
    return np.array(
        np.frombuffer(out.stdout[:n * frame_bytes], dtype=np.uint8).reshape(n, h, w, 3))


class Preprocessor:
    def __init__(self, model_dir: str):
        from transformers import AutoImageProcessor
        self.image = AutoImageProcessor.from_pretrained(model_dir)
        self.video = None
        self.video_fps, self.video_frames = 2.0, 96
        try:
            from transformers import AutoVideoProcessor
            self.video = AutoVideoProcessor.from_pretrained(model_dir)
            self.video_fps = float(getattr(self.video, "fps", None) or 2.0)
            self.video_frames = int(getattr(self.video, "num_frames", None) or 96)
        except Exception:
            # Video is optional; an image-only deployment should still start.
            self.video = None
        self.cache = MediaCache()

    def image_part(self, url: str) -> Media:
        raw = fetch(url)

        def make() -> Media:
            from PIL import Image
            img = Image.open(io.BytesIO(raw))
            img.load()
            out = self.image(images=[img.convert("RGB")], return_tensors="np")
            px = np.asarray(out["pixel_values"], dtype=np.float64)
            grid = tuple(int(x) for x in np.asarray(out["image_grid_thw"]).reshape(-1)[:3])
            return Media("image", px, grid)

        return self.cache.get_or_make(raw, make)

    def video_part(self, url: str) -> Media:
        if self.video is None:
            raise MediaError("this transformers build has no MuseGlimmerVideoProcessor")
        raw = fetch(url)

        def make() -> Media:
            frames = decode_video(raw, fps=self.video_fps, max_frames=self.video_frames)
            out = self.video(videos=[frames], return_tensors="np", do_sample_frames=False)
            px = np.asarray(out["pixel_values_videos"], dtype=np.float64)
            grid = tuple(int(x) for x in np.asarray(out["video_grid_thw"]).reshape(-1)[:3])
            return Media("video", px, grid)

        return self.cache.get_or_make(raw, make)

    def frames_part(self, urls: list[str]) -> Media:
        """A video given as an explicit list of frame images.

        Some clients send frames rather than a container, and it is also the
        only video path that needs no decoder at all — so it is the one the
        tests use."""
        if self.video is None:
            raise MediaError("this transformers build has no MuseGlimmerVideoProcessor")
        from PIL import Image
        raws = [fetch(u) for u in urls]

        def make() -> Media:
            arr = []
            for r in raws:
                img = Image.open(io.BytesIO(r))
                img.load()
                arr.append(np.asarray(img.convert("RGB"), dtype=np.uint8))
            frames = np.stack(arr)
            out = self.video(videos=[frames], return_tensors="np", do_sample_frames=False)
            px = np.asarray(out["pixel_values_videos"], dtype=np.float64)
            grid = tuple(int(x) for x in np.asarray(out["video_grid_thw"]).reshape(-1)[:3])
            return Media("video", px, grid)

        return self.cache.get_or_make(b"".join(raws), make)


def split_parts(messages: list, pre: Preprocessor | None) -> tuple[list, list[Media]]:
    """Replace media parts with the template's own placeholder parts and
    collect the decoded media in prompt order.

    The template renders `{"type": "image"}` as a single `<|patch|>`; the
    caller expands that to one placeholder per MERGED vision token, because
    only then does the number of embedding rows match the number of positions
    they are scattered into.
    """
    out, media = [], []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m)
            continue
        parts = []
        for p in content:
            t = (p or {}).get("type")
            if t in ("image_url", "image"):
                url = (p.get("image_url") or {}).get("url") if t == "image_url" else p.get("url")
                if not url:
                    raise MediaError("image part with no url")
                if pre is None:
                    raise MediaError("this server was started without the vision tower")
                media.append(pre.image_part(url))
                parts.append({"type": "image"})
            elif t in ("video_url", "video"):
                src = (p.get("video_url") or {}) if t == "video_url" else p
                if pre is None:
                    raise MediaError("this server was started without the vision tower")
                urls = src.get("frames") or p.get("frames")
                if urls:
                    media.append(pre.frames_part(list(urls)))
                else:
                    url = src.get("url") or p.get("url")
                    if not url:
                        raise MediaError("video part with neither url nor frames")
                    media.append(pre.video_part(url))
                parts.append({"type": "video"})
            elif t == "input_audio":
                raise MediaError("this model has no audio tower: input_audio is not supported "
                                 "(text + image/video in, text out)")
            else:
                parts.append(p)
        mm = dict(m)
        mm["content"] = parts
        out.append(mm)
    return out, media


def expand_placeholders(text: str, media: list[Media], merge_unit: int) -> str:
    """One placeholder per merged vision token, in prompt order."""
    pos = 0
    for m in media:
        marker = PATCH if m.kind == "image" else VIDEO
        i = text.find(marker, pos)
        if i < 0:
            raise MediaError("rendered prompt has fewer media placeholders than media parts")
        n = m.tokens(merge_unit)
        text = text[:i] + marker * n + text[i + len(marker):]
        # Resume PAST what was just written, or the next part would find this
        # one's copies.
        pos = i + len(marker) * n
    return text
