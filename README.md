# videogames_utils

Utilities for processing and analyzing CNeuroMod videogame data, including replay file processing, psychophysics measurements, video generation, and an interactive GUI for exploring datasets.

![GUI Screenshot](src/videogames_utils/gui/resources/GUI_screenshot.png)

## Installation

```bash
pip install git+https://github.com/cneuromod/videogames_utils.git
```

Or install from source:

```bash
git clone https://github.com/cneuromod/videogames_utils.git
cd videogames_utils
pip install -e .
```

## Requirements

- Python >= 3.9
- stable-retro
- numpy
- Pillow
- scikit-video
- opencv-python

## Features

### Replay Processing

Process and analyze BK2 replay files from retro game emulators:

```python
from videogames_utils import replay_bk2

# Iterate through a replay file
for frame, keys, annotations, audio_chunk, audio_rate, truncate, actions, state in replay_bk2('replay.bk2'):
    # Process each frame
    print(f"Frame shape: {frame.shape}")
    print(f"Keys pressed: {keys}")
    print(f"Reward: {annotations['reward']}")
```

Extract complete replay data:

```python
from videogames_utils.replay import get_variables_from_replay

# Get all replay data at once
rep_vars, info, frames, states, audio, audio_rate = get_variables_from_replay(
    'replay.bk2',
    skip_first_step=True,
    game='SuperMarioBros-Nes'
)
```

### Audio Processing

Export audio from replay files:

```python
from videogames_utils.replay import write_wav

# Write audio to WAV file
write_wav(audio, audio_rate, 'output.wav')
```

### Psychophysics Analysis

Compute psychophysical measures from video and audio data:

```python
from videogames_utils.psychophysics import (
    audio_envelope_per_frame,
    compute_luminance,
    compute_optical_flow
)

# Compute audio envelope synchronized to video frames
envelope = audio_envelope_per_frame(audio, sample_rate=32040, frame_rate=60.0)

# Compute luminance for each frame
luminance = compute_luminance(frames)

# Compute optical flow between consecutive frames
flow = compute_optical_flow(frames)
```

### Video Generation

Create video files from frame sequences:

```python
from videogames_utils.video import make_mp4, make_gif, make_webp

# Create MP4 with audio
make_mp4(frames, 'output.mp4', audio=audio, sample_rate=32040, fps=60)

# Create GIF
make_gif(frames, 'output.gif')

# Create WebP animation
make_webp(frames, 'output.webp')
```

### Run Recording Generation

Generate video recordings aligned with BOLD acquisition from events files. This tool reads `*_events.tsv` files, replays the corresponding `.bk2` files, and creates a single video per run that aligns with the fMRI timing (with black frames filling gaps between game replays).

```bash
# Process all runs in a dataset (using all CPUs)
vg-generate-recording mario/ -j -1

# Process a single run
vg-generate-recording mario/sub-01/ses-001/func/sub-01_ses-001_task-mario_run-01_desc-annotated_events.tsv

# With verbose output
vg-generate-recording mario/ -v

# Save alignment reports to a directory
vg-generate-recording mario/ -j -1 --report-dir ./reports

# Without audio
vg-generate-recording mario/ -j -1 --no-audio
```

The output videos are saved in the same `func/` directory as the events files, with the naming pattern `sub-XX_ses-XXX_task-TASK_run-XX_recording.mp4`.

**Options:**
- `-j N, --jobs N`: Number of parallel workers (use `-1` for all CPUs)
- `-v, --verbose`: Show detailed logging output
- `--no-audio`: Generate video without audio
- `--report-dir DIR`: Save alignment reports to specified directory
- `--fps N`: Frames per second (default: 60)

### GUI Visualizer

An interactive tool for exploring CNeuroMod videogame datasets with synchronized visualization of gameplay, brain activity, and physiological signals.

```bash
vg-visualizer
```

Features:
- **Replay Browser**: Browse datasets (mario, shinobi, mario3, mariostars) and select replays
- **Video Playback**: Play, pause, forward, backward controls with frame slider
- **Controller Display**: Visual representation of button presses in real-time
- **Game Variables**: Display variables as scrolling timeseries
- **Brain Activity Visualization**: Glass brain plots showing parcellated brain activity
- **Physiological Data Visualization**: Scrolling physio timeseries display (PPG, ECG, RSP, EDA)

For detailed usage instructions, see [GUI_README.md](GUI_README.md).

## Module Overview

### `replay.py`

Core replay functionality for processing BK2 files:

- `replay_bk2()` - Iterator for stepping through replay files
- `get_variables_from_replay()` - Extract complete replay data
- `reformat_info()` - Parse replay metadata from filenames
- `assemble_audio()` - Concatenate audio chunks
- `write_wav()` - Export audio to WAV format

### `psychophysics.py`

Psychophysical measurements for video game stimuli:

- `audio_envelope_per_frame()` - RMS-based audio envelope at frame rate
- `compute_luminance()` - Mean luminance per frame
- `compute_optical_flow()` - Frame-to-frame motion analysis

### `video.py`

Video file generation utilities:

- `make_mp4()` - Create MP4 files with optional audio
- `make_gif()` - Create GIF animations
- `make_webp()` - Create WebP animations

### `generate_run_recording.py`

Generate BOLD-aligned video recordings from events files:

- `generate_aligned_recording()` - Main function to create aligned recordings
- `find_all_events_files()` - Find events files in a dataset
- `StreamingVideoWriter` - Memory-efficient video writer class

```python
from videogames_utils import generate_aligned_recording

# Generate a recording for a single run
report = generate_aligned_recording(
    events_path="mario/sub-01/ses-001/func/sub-01_ses-001_task-mario_run-01_desc-annotated_events.tsv",
    fps=60,
    include_audio=True
)
```

## CNeuroMod Integration

This package is designed for the CNeuroMod project's videogame data. Replay filenames are expected to follow the CNeuroMod naming convention:

```
sub-{subject}_ses-{session}_level-{level}_*.bk2
```

Metadata is automatically extracted from filenames using this convention.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Authors

CNeuroMod
