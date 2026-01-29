"""
Events display widget for annotated events
"""

from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from .utils import load_annotated_events


class EventsWidget(QWidget):
    """Widget for displaying annotated events"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.events_df = pd.DataFrame()
        self.fps = 60
        self.current_time = 0.0
        self.active_events = []
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        events_group = QGroupBox("Active Events")
        events_layout = QVBoxLayout()
        events_layout.setContentsMargins(4, 4, 4, 4)

        self.events_list = QListWidget()
        self.events_list.setMaximumHeight(80)  # Reduced height
        events_layout.addWidget(self.events_list)

        self.status_label = QLabel("No events loaded")
        events_layout.addWidget(self.status_label)

        events_group.setLayout(events_layout)
        layout.addWidget(events_group)

        self.setLayout(layout)

    def load_events(self, events_path: Path, bk2_filename: str, fps: int = 60):
        """
        Load annotated events for a specific replay

        Args:
            events_path: Path to desc-annotated_events.tsv
            bk2_filename: Filename of the .bk2 replay
            fps: Frames per second for time conversion
        """
        self.fps = fps

        try:
            self.events_df = load_annotated_events(events_path, bk2_filename)

            if self.events_df.empty:
                self.status_label.setText("No events found for this replay")
            else:
                # Filter out the gym-retro_game row
                self.events_df = self.events_df[self.events_df['trial_type'] != 'gym-retro_game']
                n_events = len(self.events_df)
                self.status_label.setText(f"Loaded {n_events} events")

        except Exception as e:
            self.status_label.setText(f"Error loading events: {e}")
            print(f"Error loading events: {e}")

    def update_position(self, frame_idx: int):
        """Update active events based on current frame"""
        if self.events_df.empty:
            return

        current_time = frame_idx / self.fps

        # Find active events
        # Event is active if: onset <= current_time < onset + display_duration
        self.active_events = []

        for idx, row in self.events_df.iterrows():
            onset = row['onset']
            duration = row.get('display_duration', row.get('duration', 0))

            # Check if event is active
            if onset <= current_time < onset + duration:
                self.active_events.append({
                    'type': row['trial_type'],
                    'onset': onset,
                    'duration': duration,
                    'time_in_event': current_time - onset
                })

        # Update display
        self.update_events_display()

    def get_event_color(self, event_type: str) -> QColor:
        """
        Get color based on event type

        Args:
            event_type: The type of event

        Returns:
            QColor for the event type
        """
        # Directions (RIGHT, LEFT, UP, DOWN)
        if any(d in event_type for d in ['RIGHT', 'LEFT', 'UP', 'DOWN']):
            return QColor(255, 200, 100)  # Orange for directions
        # Damage/health loss events (Hit/something, HealthLoss)
        elif event_type.startswith('Hit/') or 'HealthLoss' in event_type:
            return QColor(255, 100, 100)  # Red for damage/health loss
        # Kill events
        elif event_type.startswith('Kill') or event_type.startswith('Kill/'):
            return QColor(255, 150, 0)  # Orange-red for kills
        # Actions in full caps (e.g., JUMP, HIT, A, B)
        elif event_type.isupper():
            return QColor(100, 200, 255)  # Blue for actions
        # Other events
        else:
            return QColor(150, 255, 150)  # Light green for other events

    def update_events_display(self):
        """Update the events list display"""
        self.events_list.clear()

        if not self.active_events:
            item = QListWidgetItem("(no active events)")
            item.setForeground(QColor(128, 128, 128))
            self.events_list.addItem(item)
            return

        for event in self.active_events:
            text = f"{event['type']} ({event['time_in_event']:.2f}s / {event['duration']:.2f}s)"
            item = QListWidgetItem(text)
            color = self.get_event_color(event['type'])
            item.setForeground(color)
            self.events_list.addItem(item)

    def get_event_markers(self) -> List[Dict]:
        """
        Get all events as markers for plotting

        Returns:
            List of dicts with 'time', 'type', 'duration' keys
        """
        if self.events_df.empty:
            return []

        markers = []
        for idx, row in self.events_df.iterrows():
            markers.append({
                'time': row['onset'],
                'type': row['trial_type'],
                'duration': row.get('display_duration', row.get('duration', 0))
            })

        return markers
