import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import skvideo.io
from PIL import Image

from .replay import write_wav


def make_gif(selected_frames, movie_fname):
    """Create a GIF file from a list of frames."""
    frame_list = [Image.fromarray(np.uint8(img), "RGB") for img in selected_frames]

    if not frame_list:
        logging.warning(f"No frames to save in {movie_fname}")
        return

    frame_list[0].save(
        movie_fname,
        save_all=True,
        append_images=frame_list[1:],
        optimize=False,
        duration=16,
        loop=0,
    )


def make_mp4(
    selected_frames: Iterable[np.ndarray],
    movie_fname: str,
    *,
    audio: np.ndarray | None = None,
    sample_rate: int | None = None,
    fps: int = 60,
) -> None:
    """Create an MP4 file from a list of frames, with optional audio multiplexing."""

    temp_dir = tempfile.mkdtemp(prefix="cneuromod_vg_utils_")
    temp_video = Path(temp_dir) / "video_only.mp4"

    writer = skvideo.io.FFmpegWriter(
        str(temp_video), inputdict={"-r": str(fps)}, outputdict={"-r": str(fps)}
    )
    for frame in selected_frames:
        im = Image.new("RGB", (frame.shape[1], frame.shape[0]), color="white")
        im.paste(Image.fromarray(frame), (0, 0))
        writer.writeFrame(np.array(im))
    writer.close()

    final_path = Path(movie_fname)

    if audio is None or sample_rate is None:
        temp_video.replace(final_path)
        os.rmdir(temp_dir)
        return

    if audio.dtype != np.int16:
        logging.info("Casting audio to int16 before muxing")
        audio = audio.astype(np.int16)

    temp_audio = Path(temp_dir) / "audio.wav"
    write_wav(audio, sample_rate, str(temp_audio))

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(temp_video),
        "-i",
        str(temp_audio),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(final_path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg binary is required to mux audio into MP4") from exc
    finally:
        try:
            temp_audio.unlink(missing_ok=True)
            temp_video.unlink(missing_ok=True)
            os.rmdir(temp_dir)
        except OSError:
            pass


def make_webp(selected_frames, movie_fname):
    """Create a WebP file from a list of frames."""
    frame_list = [Image.fromarray(np.uint8(img), "RGB") for img in selected_frames]

    if not frame_list:
        logging.warning(f"No frames to save in {movie_fname}")
        return

    frame_list[0].save(
        movie_fname,
        "WEBP",
        quality=50,
        lossless=False,
        save_all=True,
        append_images=frame_list[1:],
        duration=16,
        loop=0,
    )
