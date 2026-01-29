"""
Main window for the Video Game Replay Visualizer
"""

from pathlib import Path
from typing import Optional, Dict, Tuple

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QStatusBar, QMenuBar, QMenu, QFileDialog, QMessageBox, QTabWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QIcon, QPixmap

from .file_browser import FileBrowser
from .video_player import VideoPlayer
from .timeseries_widget import TimeseriesWidget
from .events_widget import EventsWidget
from .glassbrain_widget import GlassBrainWidget
from .physio_widget import PhysioWidget
from .controller_widget import ControllerWidget
from .utils import detect_game_from_filename, find_annotated_events_for_replay
import pandas as pd


class ReplayVisualizerApp(QMainWindow):
    """Main application window"""

    def __init__(self, n_jobs=1):
        super().__init__()
        self.n_jobs = n_jobs  # Number of workers for brain precomputation
        self.current_replay_info = None
        self.current_dataset_path = None
        self.current_timeseries_path = None
        self.current_atlas_path = None
        self.current_run_info = None  # (session, run, onset_time)
        
        # Throttling for heavy widget updates (glassbrain, physio)
        self._last_heavy_update_frame = -1
        self._heavy_update_interval = 3  # Update every 3 frames (~20Hz for heavy widgets)
        
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("CNeuromod Videogame Replay Visualizer")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(800, 600)
        
        # Set window icon
        icon_path = Path(__file__).parent / "resources" / "logo_neuromod_small.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Create menu bar
        self.create_menu_bar()

        # Create main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()

        # Left panel: File browser
        self.file_browser = FileBrowser()
        self.file_browser.replay_selected.connect(self.on_replay_selected)
        self.file_browser.setMaximumWidth(280)
        main_layout.addWidget(self.file_browser)

        # Right panel: Video and visualizations
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: Video player + Controller + Events (horizontal layout)
        from PyQt6.QtWidgets import QSizePolicy
        top_widget = QWidget()
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_widget.setLayout(top_layout)

        # Video player (left)
        self.video_player = VideoPlayer()
        self.video_player.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.video_player.frame_changed.connect(self.on_frame_changed)
        self.video_player.button_list_changed.connect(self.on_button_list_changed)
        self.video_player.button_states_changed.connect(self.on_button_states_changed)
        top_layout.addWidget(self.video_player, stretch=2)

        # Right column: Controller (top) + Events (bottom)
        right_column = QWidget()
        right_column_layout = QVBoxLayout()
        right_column_layout.setContentsMargins(0, 0, 0, 0)
        right_column.setLayout(right_column_layout)

        # Controller row: controller + logo
        controller_row = QWidget()
        controller_row_layout = QHBoxLayout()
        controller_row_layout.setContentsMargins(0, 0, 0, 0)
        controller_row.setLayout(controller_row_layout)

        self.controller_widget = ControllerWidget()
        controller_row_layout.addWidget(self.controller_widget)

        # Add logo to the right of controller
        from PyQt6.QtWidgets import QLabel
        logo_label = QLabel()
        logo_path = Path(__file__).parent / "resources" / "logo_neuromod_small.png"
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            # Scale to reasonable size while keeping aspect ratio
            scaled_pixmap = pixmap.scaledToHeight(80, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
        controller_row_layout.addWidget(logo_label)

        right_column_layout.addWidget(controller_row)

        self.events_widget = EventsWidget()
        right_column_layout.addWidget(self.events_widget)

        top_layout.addWidget(right_column, stretch=1)
        right_splitter.addWidget(top_widget)

        # Middle: Glassbrain (left) + Physio (right) in horizontal splitter
        brain_physio_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.glassbrain_widget = GlassBrainWidget(n_jobs=self.n_jobs)
        brain_physio_splitter.addWidget(self.glassbrain_widget)
        
        self.physio_widget = PhysioWidget()
        brain_physio_splitter.addWidget(self.physio_widget)
        
        # Set initial stretch factors for glassbrain/physio (2:1 ratio)
        brain_physio_splitter.setStretchFactor(0, 2)  # Glassbrain - twice as wide
        brain_physio_splitter.setStretchFactor(1, 1)  # Physio - half the width
        
        right_splitter.addWidget(brain_physio_splitter)

        # Bottom: Timeseries
        self.timeseries_widget = TimeseriesWidget()
        right_splitter.addWidget(self.timeseries_widget)

        # Set initial stretch factors
        right_splitter.setStretchFactor(0, 3)  # Video + controller + events
        right_splitter.setStretchFactor(1, 2)  # Glassbrain + Physio
        right_splitter.setStretchFactor(2, 1)  # Timeseries

        main_layout.addWidget(right_splitter)

        central_widget.setLayout(main_layout)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Click 'Browse for Dataset...' to begin")

    def create_menu_bar(self):
        """Create the menu bar"""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open Dataset...", self)
        open_action.triggered.connect(self.open_dataset)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menu_bar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def open_dataset(self):
        """Open a dataset directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Directory",
            str(Path.home() / "DATA" / "neuromod" / "vg_data"),
            QFileDialog.Option.ShowDirsOnly
        )

        if directory:
            self.file_browser.load_dataset(Path(directory))
            self.status_bar.showMessage(f"Opened: {directory}")

    def on_replay_selected(self, replay_info: Dict):
        """Handle replay selection"""
        self.status_bar.showMessage(f"Loading replay: {replay_info['filename']}")

        self.current_replay_info = replay_info
        self.current_dataset_path = replay_info['dataset_path']

        # Load video
        try:
            self.video_player.load_replay(replay_info, self.current_dataset_path)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Replay",
                f"Failed to load replay:\n{e}"
            )
            return

        # Load variables
        variables_path = replay_info['path'].parent / (replay_info['path'].stem + '_variables.json')
        if variables_path.exists():
            try:
                self.timeseries_widget.load_variables(variables_path, fps=60)
                self.status_bar.showMessage("Variables loaded")
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Warning",
                    f"Failed to load variables:\n{e}"
                )
        else:
            QMessageBox.warning(
                self,
                "Warning",
                f"Variables file not found:\n{variables_path}"
            )

        # Load events
        # Use events_file from replay_info if available, otherwise search for it
        events_path = replay_info.get('events_file')
        if not events_path:
            events_path = self.find_annotated_events(replay_info)

        if events_path and events_path.exists():
            try:
                self.events_widget.load_events(events_path, replay_info['filename'], fps=60)
                self.status_bar.showMessage("Events loaded")
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Warning",
                    f"Failed to load events:\n{e}"
                )
                import traceback
                traceback.print_exc()
        else:
            self.events_widget.status_label.setText("No annotated events found")

        # Load brain timeseries if available
        h5_path, atlas_path = self.find_timeseries_files(replay_info)

        if h5_path and atlas_path and events_path:
            # Extract run info from events
            run_info = self.extract_run_info_from_events(events_path, replay_info['filename'])

            if run_info:
                session, run, onset_time = run_info
                
                # Get replay duration from loaded frames
                replay_duration = None
                if self.video_player.frames:
                    replay_duration = len(self.video_player.frames) / 60.0  # fps = 60
                
                try:
                    self.glassbrain_widget.load_timeseries(
                        h5_path, atlas_path, session, run, onset_time,
                        task=replay_info['task'], fps=60,
                        replay_duration=replay_duration
                    )
                    self.status_bar.showMessage("Brain timeseries loaded successfully")
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        f"Failed to load brain timeseries:\n{e}"
                    )
                    import traceback
                    traceback.print_exc()
                
                # Load physio data if available
                physio_path, physio_events_path = self.find_physio_files(replay_info, session, run)
                if physio_path and physio_path.exists():
                    try:
                        self.physio_widget.load_physio(
                            physio_path, physio_events_path,
                            onset_time, fps=60,
                            replay_duration=replay_duration
                        )
                        self.status_bar.showMessage("Physio data loaded successfully")
                    except Exception as e:
                        QMessageBox.warning(
                            self,
                            "Warning",
                            f"Failed to load physio data:\n{e}"
                        )
                        import traceback
                        traceback.print_exc()
                else:
                    self.physio_widget.clear()
            else:
                self.glassbrain_widget.clear()
                self.physio_widget.clear()
        else:
            # Clear glassbrain and physio if no data available
            self.glassbrain_widget.clear()
            self.physio_widget.clear()

        self.status_bar.showMessage("Replay loaded successfully")

    def find_annotated_events(self, replay_info: Dict) -> Optional[Path]:
        """
        Find the desc-annotated_events.tsv file for a replay

        Looks in the func/ directory for the corresponding session
        """
        return find_annotated_events_for_replay(
            replay_info['dataset_path'],
            replay_info['subject'],
            replay_info['session'],
            replay_info['task']
        )

    def find_timeseries_files(self, replay_info: Dict) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Find timeseries HDF5 and atlas NIfTI files for a subject/task

        Returns:
            (h5_path, atlas_path) or (None, None) if not found
        """
        dataset_path = replay_info['dataset_path']
        subject = replay_info['subject']
        task = replay_info['task']

        # Look for {dataset}.timeseries folder
        timeseries_root = dataset_path.parent / f"{dataset_path.name}.timeseries"

        if not timeseries_root.exists():
            return None, None

        # Look in sub-{subject}/func/
        timeseries_dir = timeseries_root / f"sub-{subject}" / "func"

        if not timeseries_dir.exists():
            return None, None

        # Look for Schaefer atlas files (matching the example pattern)
        h5_pattern = f"sub-{subject}_task-{task}_*Schaefer*timeseries.h5"
        atlas_pattern = f"sub-{subject}_task-{task}_*Schaefer*dseg.nii.gz"

        h5_files = list(timeseries_dir.glob(h5_pattern))
        atlas_files = list(timeseries_dir.glob(atlas_pattern))

        if h5_files and atlas_files:
            return h5_files[0], atlas_files[0]

        return None, None

    def find_physio_files(self, replay_info: Dict, session: str, run: int) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Find physiological data files for a subject/session/run

        Args:
            replay_info: Replay info dict with dataset_path, subject, task
            session: Session ID (e.g., '001')
            run: Run number (e.g., 1)

        Returns:
            (physio_path, events_path) or (None, None) if not found
        """
        dataset_path = replay_info['dataset_path']
        subject = replay_info['subject']
        task = replay_info['task']

        # Look for {dataset}.physprep folder
        physprep_root = dataset_path.parent / f"{dataset_path.name}.physprep"

        if not physprep_root.exists():
            return None, None

        # Look in sub-{subject}/ses-{session}/func/
        physio_dir = physprep_root / f"sub-{subject}" / f"ses-{session}" / "func"

        if not physio_dir.exists():
            return None, None

        # Look for physio files matching the run
        physio_pattern = f"sub-{subject}_ses-{session}_task-{task}_run-{run:02d}_desc-preproc_physio.tsv.gz"
        events_pattern = f"sub-{subject}_ses-{session}_task-{task}_run-{run:02d}_events.tsv"

        physio_files = list(physio_dir.glob(physio_pattern))
        events_files = list(physio_dir.glob(events_pattern))

        physio_path = physio_files[0] if physio_files else None
        events_path = events_files[0] if events_files else None

        return physio_path, events_path

    def extract_run_info_from_events(self, events_path: Path, bk2_filename: str) -> Optional[Tuple]:
        """
        Extract run number and onset time for a replay from events file

        Returns:
            (session, run_number, onset_time) or None if not found
        """
        if not events_path or not events_path.exists():
            return None

        try:
            df = pd.read_csv(events_path, sep='\t')

            # Find the gym-retro_game row that references this .bk2 file
            game_row_mask = (df['trial_type'] == 'gym-retro_game') & \
                           (df['stim_file'].str.contains(bk2_filename, na=False))

            if not game_row_mask.any():
                return None

            game_row_idx = df[game_row_mask].index[0]
            onset_time = df.loc[game_row_idx, 'onset']

            # Extract run number from events filename
            # e.g., sub-01_ses-001_task-mario_run-01_desc-annotated_events.tsv
            filename = events_path.name
            parts = filename.split('_')
            session = None
            run = None

            for part in parts:
                if part.startswith('ses-'):
                    session = part.split('-')[1]
                elif part.startswith('run-'):
                    run = int(part.split('-')[1])

            if session and run and onset_time is not None:
                return session, run, onset_time

            return None

        except Exception as e:
            print(f"Error extracting run info: {e}")
            import traceback
            traceback.print_exc()
            return None

    def on_frame_changed(self, frame_idx: int):
        """Handle frame change from video player
        
        Lightweight updates (timeseries, events) happen every frame.
        Heavy updates (glassbrain, physio) are throttled during playback,
        but always update immediately on manual seeks (slider scrubbing).
        """
        # Lightweight updates - every frame
        self.timeseries_widget.update_position(frame_idx)
        self.events_widget.update_position(frame_idx)

        # Detect manual seek: if frame jump is larger than normal playback would produce
        # Normal playback advances by 1-3 frames at a time; larger jumps indicate seeking
        frame_delta = abs(frame_idx - self._last_heavy_update_frame)
        is_seeking = frame_delta > self._heavy_update_interval * 2
        
        # Heavy updates - throttled during playback, immediate on seek
        if is_seeking or frame_delta >= self._heavy_update_interval:
            self._last_heavy_update_frame = frame_idx
            
            # Update glassbrain
            self.glassbrain_widget.update_position(frame_idx)

            # Update physio
            self.physio_widget.update_position(frame_idx)

            # Update status bar
            time_seconds = frame_idx / 60.0  # Assuming 60 FPS
            self.status_bar.showMessage(f"Frame: {frame_idx} | Time: {time_seconds:.2f}s")

    def on_button_list_changed(self, button_list):
        """Handle button list change from video player"""
        self.controller_widget.set_buttons(button_list)

    def on_button_states_changed(self, button_states):
        """Handle button states change from video player"""
        self.controller_widget.update_button_states(button_states)

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About VG Replay Visualizer",
            "Video Game Replay Visualizer\n\n"
            "Interactive tool for exploring CNeuroMod videogame datasets\n\n"
            "Features:\n"
            "- Replay .bk2 game recordings\n"
            "- Display game state variables as timeseries\n"
            "- Show annotated events\n"
            "- Visualize brain activity with glass brain plots\n"
            "- Synchronized playback controls\n\n"
            "Part of the videogames_utils toolbox"
        )
