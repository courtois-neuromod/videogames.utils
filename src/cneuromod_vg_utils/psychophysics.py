# Psychophysics utilities for the video game data
import cv2
import numpy as np


def audio_envelope_per_frame(
    audio: np.ndarray,
    sample_rate: int,
    frame_rate: float = 60.0,
    frame_count: int | None = None,
) -> np.ndarray:
    """Compute an RMS-based audio envelope sampled at the video frame rate."""

    audio = np.asarray(audio)
    if audio.size == 0:
        return np.empty(0, dtype=np.float32)

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if frame_rate <= 0:
        raise ValueError("frame_rate must be positive")
    if frame_count is not None and frame_count <= 0:
        raise ValueError("frame_count must be positive when provided")

    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    elif audio.ndim != 2:
        raise ValueError("audio must be a 1D or 2D array")

    if frame_count is None:
        samples_per_frame = max(int(round(sample_rate / frame_rate)), 1)
        num_frames = int(np.ceil(audio.shape[0] / samples_per_frame))
        pad_width = num_frames * samples_per_frame - audio.shape[0]
        if pad_width:
            audio = np.pad(audio, ((0, pad_width), (0, 0)))

        framed = audio.reshape(num_frames, samples_per_frame, audio.shape[1])
        rms = np.sqrt(np.mean(np.square(framed.astype(np.float32)), axis=1))
        envelope = np.mean(rms, axis=1)
        return envelope.astype(np.float32)

    frame_edges = np.linspace(0, audio.shape[0], frame_count + 1, endpoint=True)
    frame_edges = np.round(frame_edges).astype(int)
    frame_edges[-1] = audio.shape[0]
    frame_edges = np.maximum.accumulate(frame_edges)
    envelope = np.zeros(frame_count, dtype=np.float32)
    audio_f32 = audio.astype(np.float32)
    for idx in range(frame_count):
        start, stop = frame_edges[idx], frame_edges[idx + 1]
        segment = audio_f32[start:stop]
        if segment.size == 0:
            envelope[idx] = 0.0
            continue
        rms = np.sqrt(np.mean(np.square(segment), axis=0))
        envelope[idx] = float(rms.mean())
    return envelope

def compute_luminance(frames_list):
    """
    Compute the luminance of a list of video frames.

    Parameters:
    - frames_list: List of frames (numpy arrays).

    Returns:
    - A numpy array containing the luminance values for each frame.
    """
    luminance_values = []
    for frame in frames_list:
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Compute luminance (average of grayscale values)
        luminance = np.mean(gray_frame)
        luminance_values.append(luminance)
    return np.array(luminance_values)

def compute_optical_flow(frames_list):
    """
    Compute the optical flow between consecutive frames in a list.

    Parameters:
    - frames_list: List of frames (numpy arrays).

    Returns:
    - A list of optical flow magnitudes between consecutive frames.
    """
    optical_flows = []
    optical_flows.append(0.0)  # No flow for the first frame
    for i in range(1, len(frames_list)):
        prev_frame = cv2.cvtColor(frames_list[i-1], cv2.COLOR_RGB2GRAY)
        curr_frame = cv2.cvtColor(frames_list[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        optical_flows.append(np.mean(magnitude))
    return optical_flows
