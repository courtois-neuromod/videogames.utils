"""
Generate a video recording aligned with BOLD acquisition from events.tsv and bk2 files.

This script reads an events.tsv file, finds all gym-retro_game trials, generates
video from each corresponding .bk2 file, and creates a single video that aligns
with the BOLD recording timing (with black frames filling gaps between game replays).
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import stable_retro as retro
from stable_retro.enums import State
from tqdm import tqdm

from .replay import replay_bk2, assemble_audio, write_wav
from .gui.utils import detect_game_from_filename, find_rom_integration_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_events_file(input_path: Path) -> Path:
    """
    Find a single events.tsv file from the input path.

    Args:
        input_path: Path to either an events.tsv file directly, or a func directory

    Returns:
        Path to the events.tsv file

    Raises:
        FileNotFoundError: If no events file can be found
    """
    if input_path.is_file() and input_path.name.endswith('_events.tsv'):
        return input_path

    if input_path.is_dir():
        # Look for *desc-annotated_events.tsv files in current directory
        events_files = list(input_path.glob('*desc-annotated_events.tsv'))
        if events_files:
            if len(events_files) > 1:
                logger.warning(f"Multiple events files found, using: {events_files[0]}")
            return events_files[0]

        # Try broader pattern in current directory
        events_files = list(input_path.glob('*_events.tsv'))
        if events_files:
            if len(events_files) > 1:
                logger.warning(f"Multiple events files found, using: {events_files[0]}")
            return events_files[0]

    raise FileNotFoundError(f"No events.tsv file found at {input_path}")


def _extract_bids_sort_key(path: Path) -> Tuple[int, int, int]:
    """
    Extract subject, session, and run numbers for sorting.

    Returns tuple of (subject_num, session_num, run_num) for proper ordering.
    """
    path_str = str(path)

    # Extract subject number
    sub_match = re.search(r'sub-(\d+)', path_str)
    sub_num = int(sub_match.group(1)) if sub_match else 0

    # Extract session number
    ses_match = re.search(r'ses-(\d+)', path_str)
    ses_num = int(ses_match.group(1)) if ses_match else 0

    # Extract run number
    run_match = re.search(r'run-(\d+)', path_str)
    run_num = int(run_match.group(1)) if run_match else 0

    return (sub_num, ses_num, run_num)


def find_all_events_files(input_path: Path) -> List[Path]:
    """
    Find all events.tsv files from the input path.

    Handles three cases:
    1. Direct path to an events.tsv file
    2. Path to a func directory containing events files
    3. Path to a dataset root - searches sub*/ses*/func/

    Args:
        input_path: Path to events file, func directory, or dataset root

    Returns:
        List of absolute paths to events.tsv files, sorted by subject/session/run

    Raises:
        FileNotFoundError: If no events files can be found
    """
    # Make absolute without following symlinks
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    input_path = Path(os.path.normpath(input_path))

    if input_path.is_file() and input_path.name.endswith('_events.tsv'):
        return [input_path]

    if input_path.is_dir():
        # First, check if this is a func directory with events files
        events_files = list(input_path.glob('*desc-annotated_events.tsv'))
        if events_files:
            return sorted(events_files, key=_extract_bids_sort_key)

        events_files = list(input_path.glob('*_events.tsv'))
        if events_files:
            return sorted(events_files, key=_extract_bids_sort_key)

        # Search recursively in sub*/ses*/func/ pattern (dataset root)
        events_files = list(input_path.glob('sub-*/ses-*/func/*desc-annotated_events.tsv'))
        if events_files:
            return sorted(events_files, key=_extract_bids_sort_key)

        events_files = list(input_path.glob('sub-*/ses-*/func/*_events.tsv'))
        if events_files:
            return sorted(events_files, key=_extract_bids_sort_key)

    raise FileNotFoundError(
        f"No events.tsv files found at {input_path}\n"
        f"Expected pattern: {{dataset}}/sub-*/ses-*/func/*_events.tsv"
    )


def get_dataset_path(events_path: Path) -> Path:
    """
    Determine the dataset root path from an events file path.

    Args:
        events_path: Path to the events.tsv file

    Returns:
        Path to the dataset root (absolute, without following symlinks)
    """
    # Events file is typically at: {dataset}/sub-XX/ses-XXX/func/*events.tsv
    # Go up 4 levels to get dataset root
    # Use Path.absolute() instead of resolve() to avoid following git-annex symlinks
    if not events_path.is_absolute():
        events_path = Path.cwd() / events_path
    # Normalize the path (resolve .. and .) without following symlinks
    events_path = Path(os.path.normpath(events_path))

    # Go up: func -> ses-XXX -> sub-XX -> dataset
    current = events_path.parent  # func/
    current = current.parent      # ses-XXX/
    current = current.parent      # sub-XX/
    current = current.parent      # dataset/

    return current


def get_bold_duration(events_path: Path, events_df: pd.DataFrame) -> float:
    """
    Determine the total BOLD duration for the run.

    First tries to read from BOLD JSON sidecar file, then falls back to
    the last event's onset + duration.

    Args:
        events_path: Path to the events file
        events_df: DataFrame with events data

    Returns:
        Total duration in seconds
    """
    # Try to find BOLD JSON sidecar
    bold_json_pattern = events_path.name.replace('_desc-annotated_events.tsv', '_bold.json')
    bold_json_pattern = bold_json_pattern.replace('_events.tsv', '_bold.json')
    bold_json_path = events_path.parent / bold_json_pattern

    if bold_json_path.exists():
        try:
            with open(bold_json_path) as f:
                bold_info = json.load(f)
            if 'RepetitionTime' in bold_info and 'NumVolumes' in bold_info:
                duration = bold_info['RepetitionTime'] * bold_info['NumVolumes']
                logger.info(f"BOLD duration from JSON: {duration:.2f}s")
                return duration
        except Exception as e:
            logger.warning(f"Could not read BOLD JSON: {e}")

    # Fall back to last event's onset + duration
    last_event = events_df.iloc[-1]
    duration = last_event['onset'] + last_event.get('duration', 0)
    logger.info(f"BOLD duration from last event: {duration:.2f}s")
    return duration


def parse_game_trials(events_df: pd.DataFrame) -> List[Dict]:
    """
    Parse all gym-retro_game trials from the events DataFrame.

    Args:
        events_df: DataFrame with events data

    Returns:
        List of dicts with trial info: onset, duration, stim_file, bk2_filename
    """
    game_mask = events_df['trial_type'] == 'gym-retro_game'
    game_rows = events_df[game_mask]

    trials = []
    for _, row in game_rows.iterrows():
        if pd.isna(row.get('stim_file')):
            logger.warning(f"Skipping trial at onset {row['onset']} - no stim_file")
            continue

        # Extract just the filename from stim_file (in case path format differs)
        stim_file = row['stim_file']
        bk2_filename = Path(stim_file).name

        trials.append({
            'onset': float(row['onset']),
            'duration': float(row['duration']) if not pd.isna(row.get('duration')) else 0,
            'stim_file': stim_file,
            'bk2_filename': bk2_filename
        })

    return trials


def find_bk2_file(dataset_path: Path, bk2_filename: str, events_path: Path) -> Optional[Path]:
    """
    Find a bk2 file in the dataset.

    Searches in the gamelogs directory for the subject/session from the events file.

    Args:
        dataset_path: Path to the dataset root
        bk2_filename: Name of the bk2 file to find
        events_path: Path to the events file (to extract subject/session)

    Returns:
        Path to the bk2 file, or None if not found
    """
    # Extract subject and session from events filename
    # e.g., sub-01_ses-001_task-mario_run-01_desc-annotated_events.tsv
    events_name = events_path.name
    parts = events_name.split('_')
    subject = None
    session = None

    for part in parts:
        if part.startswith('sub-'):
            subject = part
        elif part.startswith('ses-'):
            session = part

    if subject and session:
        # Look in the specific gamelogs directory
        gamelogs_dir = dataset_path / subject / session / 'gamelogs'
        bk2_path = gamelogs_dir / bk2_filename
        if bk2_path.exists():
            return bk2_path

    # Fallback: search all gamelogs directories
    matches = list(dataset_path.glob(f'sub-*/ses-*/gamelogs/{bk2_filename}'))
    if matches:
        return matches[0]

    return None


class StreamingVideoWriter:
    """
    Memory-efficient video writer that writes frames incrementally.

    Uses OpenCV for video writing and ffmpeg for audio muxing.
    """

    def __init__(self, output_path: Path, fps: int = 60, frame_shape: Optional[Tuple[int, int]] = None):
        # Ensure output path is absolute
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        self.output_path = Path(os.path.normpath(output_path))
        self.fps = fps
        self.frame_shape = frame_shape  # (height, width)
        self.writer = None
        self.temp_video_path = None
        self.frame_count = 0

    def _init_writer(self, frame: np.ndarray):
        """Initialize the video writer with the first frame's dimensions."""
        height, width = frame.shape[:2]
        self.frame_shape = (height, width)

        # Use temp file for video without audio (same directory as output)
        self.temp_video_path = self.output_path.with_suffix('.temp.mp4')
        logger.debug(f"Temp video path: {self.temp_video_path}")
        logger.debug(f"Final output path: {self.output_path}")

        # Use mp4v codec for compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            str(self.temp_video_path),
            fourcc,
            self.fps,
            (width, height)
        )

    def write_frame(self, frame: np.ndarray):
        """Write a single frame to the video."""
        if self.writer is None:
            self._init_writer(frame)

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame_bgr)
        self.frame_count += 1

    def write_black_frames(self, num_frames: int, audio_chunks: Optional[List[np.ndarray]] = None,
                           audio_rate: Optional[int] = None):
        """Write black frames for padding, with optional silent audio padding.

        Args:
            num_frames: Number of black frames to write
            audio_chunks: List to append silence chunks to (if audio is being collected)
            audio_rate: Audio sample rate (needed to calculate silence duration)
        """
        if self.frame_shape is None:
            raise ValueError("Cannot write black frames before frame shape is known")

        black_frame = np.zeros((*self.frame_shape, 3), dtype=np.uint8)

        # Calculate silence samples if audio is being collected
        if audio_chunks is not None and audio_rate is not None:
            # Calculate total silence duration for all black frames
            silence_duration = num_frames / self.fps
            silence_samples = int(silence_duration * audio_rate)
            # Stereo silence (2 channels) - shape must be (samples, 2) to match game audio
            silence = np.zeros((silence_samples, 2), dtype=np.int16)
            audio_chunks.append(silence)

        for _ in range(num_frames):
            self.writer.write(black_frame)
            self.frame_count += 1

    def finalize(self, audio_chunks: Optional[List[np.ndarray]] = None, audio_rate: Optional[int] = None):
        """Finalize the video, optionally adding audio."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        if not self.temp_video_path or not self.temp_video_path.exists():
            logger.error(f"Temp video file not found: {self.temp_video_path}")
            return

        logger.debug(f"Finalizing video: {self.temp_video_path} -> {self.output_path}")

        if audio_chunks and audio_rate:
            # Mux audio with video using ffmpeg
            audio_data = assemble_audio(audio_chunks)
            self._mux_audio(audio_data, audio_rate)
        else:
            # Just rename temp file to final output
            self._rename_temp_to_output()

    def _rename_temp_to_output(self):
        """Rename temp file to final output path."""
        if not self.temp_video_path or not self.temp_video_path.exists():
            logger.error(f"Temp file does not exist: {self.temp_video_path}")
            return

        try:
            # Ensure parent directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Remove existing output file if it exists
            if self.output_path.exists():
                self.output_path.unlink()

            self.temp_video_path.rename(self.output_path)
            logger.debug(f"Renamed {self.temp_video_path} to {self.output_path}")
        except Exception as e:
            logger.error(f"Failed to rename temp file: {e}")
            # Try copy instead of rename (in case of cross-device rename)
            try:
                shutil.move(str(self.temp_video_path), str(self.output_path))
                logger.debug(f"Moved {self.temp_video_path} to {self.output_path}")
            except Exception as e2:
                logger.error(f"Failed to move temp file: {e2}")

    def _mux_audio(self, audio_data: np.ndarray, audio_rate: int):
        """Mux audio with video using ffmpeg."""
        # Write audio to temp file
        temp_dir = tempfile.mkdtemp(prefix="vg_recording_")
        temp_audio = Path(temp_dir) / "audio.wav"

        try:
            write_wav(audio_data, audio_rate, str(temp_audio))

            # Ensure parent directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Use ffmpeg to mux
            cmd = [
                'ffmpeg', '-y', '-loglevel', 'error',
                '-i', str(self.temp_video_path),
                '-i', str(temp_audio),
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                str(self.output_path)
            ]

            logger.debug(f"Running ffmpeg: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.warning(f"ffmpeg failed: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd)

            # Clean up temp video
            if self.temp_video_path.exists():
                self.temp_video_path.unlink()

            logger.debug(f"Successfully muxed audio to {self.output_path}")

        except subprocess.CalledProcessError as e:
            # If ffmpeg fails, just use video without audio
            logger.warning(f"Failed to mux audio, saving video without audio: {e}")
            self._rename_temp_to_output()
        except Exception as e:
            logger.error(f"Error during audio muxing: {e}")
            self._rename_temp_to_output()
        finally:
            # Clean up temp audio
            if temp_audio.exists():
                temp_audio.unlink()
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass


def iterate_replay_frames(
    bk2_path: Path,
    dataset_path: Path,
    skip_first_step: bool = False
) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray], int]]:
    """
    Generator that yields frames from a bk2 file one at a time.

    Memory efficient - doesn't store all frames in memory.

    Args:
        bk2_path: Path to the .bk2 file
        dataset_path: Path to the dataset root (for finding ROM)
        skip_first_step: Whether to skip the first step of the replay

    Yields:
        Tuple of (frame, audio_chunk or None, audio_rate)
    """
    # Ensure absolute paths for stable_retro (without following symlinks)
    if not bk2_path.is_absolute():
        bk2_path = Path.cwd() / bk2_path
    bk2_path = Path(os.path.normpath(bk2_path))

    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path
    dataset_path = Path(os.path.normpath(dataset_path))

    game_type = detect_game_from_filename(bk2_path.name)
    rom_integration_path = find_rom_integration_path(dataset_path, game_type)

    if not rom_integration_path:
        raise FileNotFoundError(
            f"ROM integration directory not found for {game_type}.\n"
            f"Expected in: {dataset_path / 'stimuli'}"
        )

    # Add ROM integration path to stable_retro's search paths (must be absolute)
    stimuli_path = rom_integration_path.parent
    if not stimuli_path.is_absolute():
        stimuli_path = Path.cwd() / stimuli_path
    retro.data.Integrations.add_custom_path(str(Path(os.path.normpath(stimuli_path))))

    logger.info(f"Replaying {bk2_path.name} (skip_first_step={skip_first_step})")

    replay_gen = replay_bk2(
        str(bk2_path),
        skip_first_step=skip_first_step,
        state=State.NONE,
        game=None,
        scenario=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY
    )

    for frame, keys, annotations, audio_chunk, chunk_rate, truncate, actions, state in replay_gen:
        yield frame, audio_chunk if audio_chunk.size else None, chunk_rate


def generate_aligned_recording(
    events_path: Path,
    output_path: Optional[Path] = None,
    fps: int = 60,
    include_audio: bool = True,
    show_progress: bool = True
) -> Dict:
    """
    Generate a video recording aligned with BOLD timing from events.tsv.

    Uses streaming approach to minimize memory usage - frames are written
    to disk as they are generated instead of being stored in memory.

    Args:
        events_path: Path to the events.tsv file or func directory
        output_path: Path for output video (default: same as events with _recording.mp4)
        fps: Frames per second for the video
        include_audio: Whether to include audio in the output
        show_progress: Whether to show progress bar for trials

    Returns:
        Dict with alignment report information
    """
    # Find events file and ensure absolute path
    events_path = find_events_file(events_path)
    if not events_path.is_absolute():
        events_path = Path.cwd() / events_path
    events_path = Path(os.path.normpath(events_path))
    logger.info(f"Using events file: {events_path}")

    # Determine output path - always in same directory as events file
    if output_path is None:
        # Remove _desc-annotated and _events.tsv, add _recording.mp4
        output_name = events_path.name.replace('_desc-annotated_events.tsv', '_recording.mp4')
        output_name = output_name.replace('_events.tsv', '_recording.mp4')  # fallback for non-annotated
        output_path = events_path.parent / output_name
    # Ensure output_path is absolute
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path = Path(os.path.normpath(output_path))
    logger.info(f"Output path: {output_path}")

    # Load events
    events_df = pd.read_csv(events_path, sep='\t')
    logger.info(f"Loaded {len(events_df)} events")

    # Get dataset path
    dataset_path = get_dataset_path(events_path)
    logger.info(f"Dataset path: {dataset_path}")

    # Get total BOLD duration
    total_duration = get_bold_duration(events_path, events_df)
    total_frames = int(total_duration * fps)
    logger.info(f"Total duration: {total_duration:.2f}s ({total_frames} frames)")

    # Parse game trials
    trials = parse_game_trials(events_df)
    logger.info(f"Found {len(trials)} game trials")

    if not trials:
        raise ValueError("No gym-retro_game trials found in events file")

    # Initialize alignment report
    report = {
        'events_file': str(events_path),
        'output_file': str(output_path),
        'target_duration_s': total_duration,
        'target_frames': total_frames,
        'fps': fps,
        'trials': [],
        'warnings': [],
        'errors': []
    }

    # Initialize streaming video writer
    video_writer = StreamingVideoWriter(output_path, fps=fps)
    all_audio_chunks = []
    audio_rate = None
    frame_shape = None
    current_frame = 0

    # Create trial iterator with optional progress bar
    trial_iter = enumerate(trials)
    if show_progress:
        trial_iter = tqdm(
            list(trial_iter),
            desc=f"Trials ({events_path.stem[:30]})",
            unit="trial",
            leave=False
        )

    try:
        for trial_idx, trial in trial_iter:
            onset_frame = int(trial['onset'] * fps)

            # Add black frames before this trial if needed
            if onset_frame > current_frame:
                gap_frames = onset_frame - current_frame
                logger.info(f"Adding {gap_frames} black frames before trial {trial_idx}")

                if frame_shape is None:
                    # Use default NES resolution as fallback
                    frame_shape = (240, 256)
                    logger.warning("Using default NES resolution (256x240) for black frames")
                    video_writer.frame_shape = frame_shape

                # Pad audio with silence if we're collecting audio and know the rate
                video_writer.write_black_frames(
                    gap_frames,
                    audio_chunks=all_audio_chunks if include_audio and audio_rate else None,
                    audio_rate=audio_rate
                )
                current_frame = onset_frame

            # Generate replay frames - find bk2 in gamelogs directory
            bk2_path = find_bk2_file(dataset_path, trial['bk2_filename'], events_path)

            if bk2_path is None:
                error_msg = f"BK2 file not found: {trial['bk2_filename']} in {dataset_path}/sub-*/ses-*/gamelogs/"
                logger.error(error_msg)
                report['errors'].append(error_msg)
                continue

            # First trial of the run needs skip_first_step=True
            skip_first = (trial_idx == 0)

            trial_frame_count = 0
            trial_start_frame = current_frame

            try:
                # Stream frames directly to video writer
                for frame, audio_chunk, rate in iterate_replay_frames(
                    bk2_path, dataset_path, skip_first_step=skip_first
                ):
                    video_writer.write_frame(frame)
                    trial_frame_count += 1
                    current_frame += 1

                    if frame_shape is None:
                        frame_shape = frame.shape[:2]

                    if include_audio and audio_chunk is not None:
                        all_audio_chunks.append(audio_chunk)
                        if audio_rate is None:
                            audio_rate = rate

            except Exception as e:
                error_msg = f"Failed to generate frames for {bk2_path.name}: {e}"
                logger.error(error_msg)
                report['errors'].append(error_msg)
                continue

            logger.info(f"Generated {trial_frame_count} frames from {bk2_path.name}")

            # Track trial info for report
            trial_info = {
                'trial_index': trial_idx,
                'stim_file': trial['stim_file'],
                'events_onset_s': trial['onset'],
                'events_duration_s': trial['duration'],
                'generated_frames': trial_frame_count,
                'generated_duration_s': trial_frame_count / fps,
                'video_start_frame': trial_start_frame,
                'video_start_s': trial_start_frame / fps
            }

            # Check for duration mismatch
            if trial['duration'] > 0:
                generated_duration = trial_frame_count / fps
                duration_diff = abs(generated_duration - trial['duration'])
                if duration_diff > 0.5:  # More than 0.5s difference
                    warning = (f"Trial {trial_idx}: Duration mismatch - "
                              f"events: {trial['duration']:.2f}s, "
                              f"generated: {generated_duration:.2f}s")
                    logger.warning(warning)
                    report['warnings'].append(warning)
                trial_info['duration_diff_s'] = generated_duration - trial['duration']

            report['trials'].append(trial_info)

        # Add trailing black frames if needed
        if current_frame < total_frames:
            gap_frames = total_frames - current_frame
            logger.info(f"Adding {gap_frames} trailing black frames")
            if frame_shape:
                video_writer.write_black_frames(
                    gap_frames,
                    audio_chunks=all_audio_chunks if include_audio and audio_rate else None,
                    audio_rate=audio_rate
                )
                current_frame += gap_frames

        # Update report with final stats
        report['generated_frames'] = current_frame
        report['generated_duration_s'] = current_frame / fps
        report['frame_count_match'] = current_frame == total_frames
        report['duration_diff_s'] = report['generated_duration_s'] - total_duration

        if abs(report['duration_diff_s']) > 1.0:
            warning = (f"Total duration mismatch: "
                      f"target: {total_duration:.2f}s, "
                      f"generated: {report['generated_duration_s']:.2f}s")
            logger.warning(warning)
            report['warnings'].append(warning)

        # Finalize video with audio
        logger.info(f"Finalizing video with {current_frame} frames...")
        video_writer.finalize(
            audio_chunks=all_audio_chunks if include_audio else None,
            audio_rate=audio_rate
        )

        # Verify the output file exists
        if output_path.exists():
            logger.info(f"Video saved to: {output_path}")
        else:
            logger.error(f"Video file NOT found at: {output_path}")

    except Exception as e:
        # Clean up on error
        if video_writer.writer is not None:
            video_writer.writer.release()
        if video_writer.temp_video_path and video_writer.temp_video_path.exists():
            video_writer.temp_video_path.unlink()
        raise

    return report


def format_report(report: Dict) -> str:
    """
    Format the alignment report as a human-readable string.

    Args:
        report: Dict with alignment report information

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "ALIGNMENT REPORT",
        "=" * 60,
        "",
        f"Events file: {report['events_file']}",
        f"Output file: {report['output_file']}",
        "",
        "TARGET:",
        f"  Duration: {report['target_duration_s']:.2f}s",
        f"  Frames: {report['target_frames']} @ {report['fps']} fps",
        "",
        "GENERATED:",
        f"  Duration: {report['generated_duration_s']:.2f}s",
        f"  Frames: {report['generated_frames']}",
        f"  Match: {'YES' if report['frame_count_match'] else 'NO'}",
        f"  Diff: {report['duration_diff_s']:+.4f}s",
        "",
        f"TRIALS ({len(report['trials'])}):",
    ]

    for trial in report['trials']:
        lines.append(f"  [{trial['trial_index']}] {Path(trial['stim_file']).name}")
        lines.append(f"      Events onset: {trial['events_onset_s']:.2f}s, "
                    f"Video start: {trial['video_start_s']:.2f}s")
        lines.append(f"      Events duration: {trial['events_duration_s']:.2f}s, "
                    f"Generated: {trial['generated_duration_s']:.2f}s")
        if 'duration_diff_s' in trial:
            lines.append(f"      Duration diff: {trial['duration_diff_s']:+.4f}s")

    if report['warnings']:
        lines.extend([
            "",
            "WARNINGS:",
        ])
        for warning in report['warnings']:
            lines.append(f"  - {warning}")

    if report['errors']:
        lines.extend([
            "",
            "ERRORS:",
        ])
        for error in report['errors']:
            lines.append(f"  - {error}")

    lines.extend([
        "",
        "=" * 60,
    ])

    # Summary
    if report['frame_count_match'] and not report['errors']:
        lines.append("STATUS: SUCCESS - Video aligns with events")
    elif report['errors']:
        lines.append("STATUS: FAILED - Errors occurred during generation")
    else:
        lines.append("STATUS: WARNING - Video generated but may not align perfectly")

    lines.append("=" * 60)

    return "\n".join(lines)


def generate_summary_report(
    all_reports: List[Dict],
    errors: List[Tuple[Path, str]],
    output_path: Path
) -> None:
    """
    Generate a consolidated TSV summary report for all processed files.

    Args:
        all_reports: List of report dictionaries from successful processing
        errors: List of (events_path, error_message) tuples for failed files
        output_path: Path to save the TSV report
    """
    # Build rows for TSV
    rows = []

    # Process successful reports
    for report in all_reports:
        events_file = Path(report['events_file'])
        # Extract subject/session from path
        parts = events_file.parts
        sub = next((p for p in parts if p.startswith('sub-')), '')
        ses = next((p for p in parts if p.startswith('ses-')), '')

        # Determine status
        if report['errors']:
            status = 'PARTIAL'
        elif not report['frame_count_match']:
            status = 'WARNING'
        else:
            status = 'OK'

        # Count trials with issues
        n_trials = len(report['trials'])
        trials_with_mismatch = sum(
            1 for t in report['trials']
            if abs(t.get('duration_diff_s', 0)) > 0.1
        )

        # Get first error/warning if any
        first_issue = ''
        if report['errors']:
            first_issue = report['errors'][0]
        elif report['warnings']:
            first_issue = report['warnings'][0]

        rows.append({
            'subject': sub,
            'session': ses,
            'events_file': events_file.name,
            'status': status,
            'target_duration_s': f"{report['target_duration_s']:.2f}",
            'generated_duration_s': f"{report['generated_duration_s']:.2f}",
            'duration_diff_s': f"{report['duration_diff_s']:+.3f}",
            'n_trials': n_trials,
            'trials_with_mismatch': trials_with_mismatch,
            'n_errors': len(report['errors']),
            'n_warnings': len(report['warnings']),
            'issue': first_issue[:100] if first_issue else ''
        })

    # Process failed files
    for events_path, error_msg in errors:
        parts = events_path.parts
        sub = next((p for p in parts if p.startswith('sub-')), '')
        ses = next((p for p in parts if p.startswith('ses-')), '')

        # Extract first line of error
        first_line = error_msg.split('\n')[0][:100]

        rows.append({
            'subject': sub,
            'session': ses,
            'events_file': events_path.name,
            'status': 'FAILED',
            'target_duration_s': '',
            'generated_duration_s': '',
            'duration_diff_s': '',
            'n_trials': '',
            'trials_with_mismatch': '',
            'n_errors': 1,
            'n_warnings': 0,
            'issue': first_line
        })

    # Sort by subject, session
    rows.sort(key=lambda r: (r['subject'], r['session']))

    # Write TSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)

    # Print summary statistics
    n_ok = sum(1 for r in rows if r['status'] == 'OK')
    n_warning = sum(1 for r in rows if r['status'] == 'WARNING')
    n_partial = sum(1 for r in rows if r['status'] == 'PARTIAL')
    n_failed = sum(1 for r in rows if r['status'] == 'FAILED')

    print(f"\nSummary report saved to: {output_path}")
    print(f"  OK: {n_ok}  |  WARNING: {n_warning}  |  PARTIAL: {n_partial}  |  FAILED: {n_failed}")


def process_single_events_file(
    events_path: Path,
    output_path: Optional[Path],
    fps: int,
    include_audio: bool,
    report_dir: Optional[Path],
    show_progress: bool = False
) -> Tuple[Path, Optional[Dict], Optional[str]]:
    """
    Process a single events file. Used for both sequential and parallel processing.

    Returns:
        Tuple of (events_path, report_dict or None, error_message or None)
    """
    try:
        report = generate_aligned_recording(
            events_path,
            output_path=output_path,
            fps=fps,
            include_audio=include_audio,
            show_progress=show_progress
        )

        # Save report if report_dir specified
        if report_dir:
            report_dir.mkdir(parents=True, exist_ok=True)
            report_name = events_path.stem + '_report.txt'
            report_path = report_dir / report_name
            report_text = format_report(report)
            with open(report_path, 'w') as f:
                f.write(report_text)
            # Also save JSON version
            json_path = report_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2)

        return (events_path, report, None)

    except Exception as e:
        import traceback
        return (events_path, None, f"{e}\n{traceback.format_exc()}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate video recordings aligned with BOLD acquisition from events.tsv files"
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to events.tsv file, func directory, or dataset root (sub-*/ses-*/func/)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output video path (only valid for single events file; default: same directory with _recording.mp4)"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1; use -1 for all CPUs)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second (default: 60)"
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Exclude audio from the output video"
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory to save alignment reports (default: print to stdout)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Handle -j -1 meaning all CPUs
    n_jobs = args.jobs
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    elif n_jobs < -1:
        n_jobs = max(1, multiprocessing.cpu_count() + n_jobs + 1)

    # Configure logging based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # Suppress all logging output - only show progress bar
        logging.getLogger().setLevel(logging.CRITICAL + 1)

    try:
        # Find all events files
        events_files = find_all_events_files(args.input_path)

        if args.verbose:
            print(f"Found {len(events_files)} events file(s)")

        if args.output and len(events_files) > 1:
            if args.verbose:
                print("Warning: --output ignored when processing multiple events files")

        all_reports = []
        errors = []

        if n_jobs > 1 and len(events_files) > 1:
            # Parallel processing
            if args.verbose:
                print(f"Processing with {n_jobs} parallel workers...")

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(
                        process_single_events_file,
                        events_path,
                        None,  # output_path
                        args.fps,
                        not args.no_audio,
                        args.report_dir,
                        False  # show_progress disabled in parallel mode
                    ): events_path
                    for events_path in events_files
                }

                # Process results with progress bar
                with tqdm(total=len(events_files), desc="Processing runs", unit="run") as pbar:
                    for future in as_completed(futures):
                        events_path, report, error = future.result()
                        if report:
                            all_reports.append(report)
                        if error:
                            errors.append((events_path, error))
                        pbar.update(1)
        else:
            # Sequential processing with progress bar
            output_path = args.output if len(events_files) == 1 else None

            # Disable progress bar for single file in verbose mode (logs are enough)
            disable_pbar = args.verbose and len(events_files) == 1
            file_iter = tqdm(events_files, desc="Processing runs", unit="run", disable=disable_pbar)

            for events_path in file_iter:
                events_path, report, error = process_single_events_file(
                    events_path,
                    output_path,
                    args.fps,
                    not args.no_audio,
                    args.report_dir,
                    show_progress=(len(events_files) == 1 and not args.verbose)
                )

                if report:
                    all_reports.append(report)
                    if not args.report_dir and len(events_files) == 1 and args.verbose:
                        print(format_report(report))
                if error:
                    errors.append((events_path, error))
                    if args.verbose:
                        print(f"Error processing {events_path}: {error}")

        # Print summary (always show for multiple files, or in verbose mode)
        if len(events_files) > 1 or args.verbose:
            print(f"\nSUMMARY: {len(all_reports)}/{len(events_files)} files processed successfully", end="")
            if errors:
                print(f" ({len(errors)} failed)")
                if args.verbose:
                    for events_path, error in errors:
                        print(f"  - {events_path.name}: {error.split(chr(10))[0]}")
            else:
                print()

        # Generate consolidated summary report for multiple files
        if len(events_files) > 1:
            # Determine report path - use input directory or report_dir if specified
            if args.report_dir:
                report_path = args.report_dir / "recording_summary.tsv"
                args.report_dir.mkdir(parents=True, exist_ok=True)
            else:
                # Save in the input directory
                if args.input_path.is_file():
                    report_path = args.input_path.parent / "recording_summary.tsv"
                else:
                    report_path = args.input_path / "recording_summary.tsv"
            generate_summary_report(all_reports, errors, report_path)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
