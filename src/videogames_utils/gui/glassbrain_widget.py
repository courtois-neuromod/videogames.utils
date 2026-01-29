"""
Glass brain visualization widget for displaying parcellated brain activity
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import os
import h5py
import nibabel as nib
from nilearn import plotting
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import queue
import time

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage
import cv2


def _init_worker():
    """Initialize worker process with low priority (nice value)."""
    try:
        # Set low priority (nice 19 = lowest priority on Linux)
        os.nice(19)
    except (OSError, AttributeError):
        pass  # Ignore on systems that don't support nice


def _compute_single_brain_plot(args):
    """
    Standalone function for multiprocessing - computes a single brain plot.
    Must be defined at module level for pickle.
    """
    tr_index, tr_data, atlas_data, atlas_affine, tr_duration = args
    
    # Small sleep to yield CPU time to other processes
    time.sleep(0.01)
    
    # Create 3D brain image from parcellated data
    brain_3d = np.zeros_like(atlas_data)

    # Map parcel values to brain volume
    for parcel_idx in range(len(tr_data)):
        brain_3d[atlas_data == (parcel_idx + 1)] = tr_data[parcel_idx]

    # Create NIfTI image
    brain_nifti = nib.Nifti1Image(brain_3d, atlas_affine)

    # Create 2x2 glass brain plot - larger figure for better fill
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor='#1a1a1a')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.05, hspace=0.1)
    
    # Define the 4 views for 2x2 layout
    views = [('l', 'Left Sagittal'), ('r', 'Right Sagittal'), 
             ('y', 'Coronal'), ('z', 'Axial')]
    
    for idx, (view_mode, view_title) in enumerate(views):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        # Create glass brain for this view
        display = plotting.plot_glass_brain(
            brain_nifti,
            colorbar=False,
            cmap='cold_hot',
            display_mode=view_mode,
            vmin=-2.5,
            vmax=2.5,
            axes=ax,
            plot_abs=False
        )
        ax.set_title(view_title, color='#cccccc', fontsize=10, pad=2)
    
    # Add overall title
    fig.suptitle(f'TR {tr_index} ({tr_index * tr_duration:.1f}s)', 
                 color='#cccccc', fontsize=12)

    # Convert figure to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img_array = np.asarray(buf).copy()  # Copy to avoid buffer issues

    # Convert RGBA to RGB
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    plt.close(fig)

    return tr_index, img_rgb


class GlassBrainWidget(QWidget):
    """Widget for displaying glass brain plots from parcellated timeseries"""

    def __init__(self, parent=None, n_jobs=1):
        super().__init__(parent)
        self.n_jobs = n_jobs  # Number of workers for precomputation
        self.timeseries_data = None
        self.atlas_img = None
        self.atlas_data = None  # Cached atlas data
        self.session = None
        self.run = None
        self.onset_time = 0.0  # Onset time of replay in the run
        self.replay_duration = None  # Duration of current replay
        self.fps = 60
        self.tr = 1.49  # TR in seconds

        # Cache for brain plots
        self.brain_cache = {}  # tr_index -> numpy image
        self.current_tr = -1
        self.prev_tr = -1
        self.last_displayed_brain = None  # Last displayed brain image (for smooth transitions)
        
        # TR range for current replay
        self.start_tr = 0
        self.end_tr = 0
        
        # Multiprocessing
        self.executor = None
        self.futures = {}  # tr_index -> future
        self.is_precomputing = False
        self.total_trs = 0
        
        # Timer to poll for completed futures
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._poll_futures)

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Progress bar for precomputation
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(15)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Precomputing brain plots: %p%")
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Label for displaying the brain plot
        self.brain_label = QLabel("No brain data loaded")
        self.brain_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.brain_label.setStyleSheet("background-color: #1a1a1a; color: #888;")
        self.brain_label.setMinimumHeight(180)  # Bigger brain plots

        layout.addWidget(self.brain_label)
        self.setLayout(layout)

    def load_timeseries(self, h5_path: Path, atlas_path: Path,
                       session: str, run: int, onset_time: float, task: str,
                       fps: int = 60, replay_duration: float = None):
        """
        Load timeseries data and atlas

        Args:
            h5_path: Path to HDF5 timeseries file
            atlas_path: Path to atlas NIfTI file
            session: Session ID (e.g., '001')
            run: Run number (e.g., 1)
            onset_time: Onset time of replay within the run (seconds)
            task: Task name (e.g., 'mario', 'shinobi')
            fps: Frames per second for the replay
            replay_duration: Duration of the replay in seconds (optional, for limiting TR computation)
        """
        try:
            # Stop any existing precomputation
            self._stop_precompute()
            
            # Clear cache when loading new data
            self.brain_cache.clear()
            self.current_tr = -1
            self.prev_tr = -1
            self.last_displayed_brain = None  # Track last displayed brain image

            # Store parameters
            self.session = session
            self.run = run
            self.onset_time = onset_time
            self.fps = fps
            self.replay_duration = replay_duration

            # Load timeseries from HDF5
            self.h5_path = h5_path
            dataset_key = f"ses-{session}/ses-{session}_task-{task}_run-{run}_timeseries"

            with h5py.File(h5_path, 'r') as f:
                if dataset_key not in f:
                    raise ValueError(f"Dataset key '{dataset_key}' not found in HDF5 file")
                self.timeseries_data = f[dataset_key][:]

            # Load atlas and cache the data
            self.atlas_img = nib.load(atlas_path)
            self.atlas_data = self.atlas_img.get_fdata()

            self.brain_label.setText("Brain data loaded - precomputing in background...")
            
            # Start background precomputation
            self._start_precompute()

        except Exception as e:
            self.brain_label.setText(f"Error loading brain data:\n{str(e)[:50]}")
            self.timeseries_data = None
            self.atlas_img = None
            self.atlas_data = None
            print(f"Error loading timeseries: {e}")
            import traceback
            traceback.print_exc()

    def _start_precompute(self):
        """Start background precomputation of all brain plots using multiprocessing"""
        if self.timeseries_data is None or self.atlas_data is None:
            return
        
        # Calculate which TRs are needed for this replay
        start_tr = int(self.onset_time / self.tr)
        
        if self.replay_duration is not None:
            end_time = self.onset_time + self.replay_duration
            end_tr = int(end_time / self.tr) + 2  # +2 for interpolation buffer
        else:
            end_tr = len(self.timeseries_data)
        
        # Clamp to valid range
        start_tr = max(0, start_tr)
        end_tr = min(end_tr, len(self.timeseries_data))
        
        self.start_tr = start_tr
        self.end_tr = end_tr
        self.total_trs = end_tr - start_tr
        
        # Show progress bar
        self.progress_bar.setMaximum(self.total_trs)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        
        # Create process pool executor with configured number of workers
        # Workers run at lowest priority (nice 19) to not interfere with UI
        self.executor = ProcessPoolExecutor(
            max_workers=self.n_jobs,
            initializer=_init_worker
        )
        
        # Submit only the relevant TR computations
        self.futures = {}
        atlas_affine = self.atlas_img.affine
        
        for tr_idx in range(start_tr, end_tr):
            args = (
                tr_idx,
                self.timeseries_data[tr_idx, :],
                self.atlas_data,
                atlas_affine,
                self.tr
            )
            future = self.executor.submit(_compute_single_brain_plot, args)
            self.futures[tr_idx] = future
        
        self.is_precomputing = True
        
        # Start polling timer (check every 100ms for completed futures)
        self.poll_timer.start(100)
    
    def _poll_futures(self):
        """Poll for completed futures and add results to cache"""
        if not self.futures:
            self.poll_timer.stop()
            return
        
        # Limit how many results we process per poll cycle to keep UI responsive
        max_per_cycle = 2
        processed = 0
        
        completed = []
        for tr_idx, future in self.futures.items():
            if future.done():
                try:
                    result_tr_idx, brain_img = future.result()
                    self.brain_cache[result_tr_idx] = brain_img
                    completed.append(tr_idx)
                    processed += 1
                    # Only process a few per cycle to keep UI smooth
                    if processed >= max_per_cycle:
                        break
                except Exception as e:
                    print(f"Error computing TR {tr_idx}: {e}")
                    completed.append(tr_idx)
        
        # Remove completed futures
        for tr_idx in completed:
            del self.futures[tr_idx]
        
        # Update progress bar
        cached_count = len(self.brain_cache)
        self.progress_bar.setValue(cached_count)
        self.progress_bar.setFormat(f"Precomputing brain plots: {cached_count}/{self.total_trs}")
        
        # Check if all done
        if not self.futures:
            self._on_precompute_finished()
    
    def _stop_precompute(self):
        """Stop any running precomputation"""
        self.poll_timer.stop()
        
        if self.executor is not None:
            # Cancel pending futures
            for future in self.futures.values():
                future.cancel()
            self.futures.clear()
            
            # Shutdown executor
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None
        
        self.is_precomputing = False
        self.progress_bar.hide()
    
    def _on_precompute_finished(self):
        """Handle precomputation completion"""
        self.poll_timer.stop()
        self.is_precomputing = False
        self.progress_bar.hide()
        
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None
        
        self.futures.clear()

    def update_position(self, frame_idx: int):
        """
        Update brain plot based on current frame

        Args:
            frame_idx: Current frame index in the replay
        """
        if self.timeseries_data is None or self.atlas_data is None:
            return

        try:
            # Calculate time in replay (seconds)
            time_in_replay = frame_idx / self.fps

            # Calculate time in run
            time_in_run = time_in_replay + self.onset_time

            # Calculate TR indices
            tr_index_float = time_in_run / self.tr
            current_tr = int(tr_index_float)
            next_tr = current_tr + 1

            # Check bounds
            if current_tr >= len(self.timeseries_data):
                return

            # Calculate interpolation alpha (0 = current TR, 1 = next TR)
            alpha = tr_index_float - current_tr
            
            # Track current TR for display purposes
            self.current_tr = current_tr

            # Check if current TR is in cache
            if current_tr not in self.brain_cache:
                # TR not yet computed - show placeholder or last displayed brain
                if self.last_displayed_brain is None:
                    # No brain to show yet
                    self.brain_label.setText(f"Computing TR {current_tr}...")
                    return
                else:
                    # Keep showing last brain but with updated label
                    brain_img = self.last_displayed_brain
            else:
                current_brain = self.brain_cache[current_tr]

                # Interpolate if we have a next TR in cache
                if alpha > 0.01 and next_tr < len(self.timeseries_data) and next_tr in self.brain_cache:
                    next_brain = self.brain_cache[next_tr]
                    brain_img = cv2.addWeighted(current_brain, 1 - alpha, next_brain, alpha, 0)
                else:
                    brain_img = current_brain
                
                # Store for next time
                self.last_displayed_brain = brain_img

            # Convert to QPixmap and display
            height, width, channel = brain_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(brain_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            # Scale to fit widget while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.brain_label.width(),
                self.brain_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.brain_label.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"Error updating brain plot: {e}")
            import traceback
            traceback.print_exc()

    def _create_brain_plot(self, tr_index: int) -> np.ndarray:
        """
        Create glass brain plot for a specific TR (fallback for non-cached)

        Args:
            tr_index: TR index to plot

        Returns:
            numpy array (height, width, 3) in RGB format
        """
        # Get timeseries data for this TR
        tr_data = self.timeseries_data[tr_index, :]

        # Create 3D brain image from parcellated data
        brain_3d = np.zeros_like(self.atlas_data)

        # Map parcel values to brain volume
        for parcel_idx in range(len(tr_data)):
            brain_3d[self.atlas_data == (parcel_idx + 1)] = tr_data[parcel_idx]

        # Create NIfTI image
        brain_img = nib.Nifti1Image(brain_3d, self.atlas_img.affine)

        # Create 2x2 glass brain plot - larger figure for better fill
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor='#1a1a1a')
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.05, hspace=0.1)
        
        # Define the 4 views for 2x2 layout
        views = [('l', 'Left Sagittal'), ('r', 'Right Sagittal'), 
                 ('y', 'Coronal'), ('z', 'Axial')]
        
        for idx, (view_mode, view_title) in enumerate(views):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            # Create glass brain for this view
            display = plotting.plot_glass_brain(
                brain_img,
                colorbar=False,
                cmap='cold_hot',
                display_mode=view_mode,
                vmin=-2.5,
                vmax=2.5,
                axes=ax,
                plot_abs=False
            )
            ax.set_title(view_title, color='#cccccc', fontsize=10, pad=2)
        
        # Add overall title
        fig.suptitle(f'TR {tr_index} ({tr_index * self.tr:.1f}s)', 
                     color='#cccccc', fontsize=12)

        # Convert figure to numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img_array = np.asarray(buf)

        # Convert RGBA to RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        plt.close(fig)

        return img_rgb

    def clear(self):
        """Clear the widget"""
        self._stop_precompute()
        self.timeseries_data = None
        self.atlas_img = None
        self.atlas_data = None
        self.brain_cache.clear()
        self.brain_label.setText("No brain data loaded")
        self.brain_label.setPixmap(QPixmap())
