from __future__ import annotations

import logging
import os.path as op
from typing import Iterable, List, Tuple

import numpy as np
import retro
from retro.enums import State

def replay_bk2(
    bk2_path,
    skip_first_step: bool = False,
    state: State = State.DEFAULT,
    game: str | None = None,
    scenario: str | None = None,
    inttype: retro.data.Integrations = retro.data.Integrations.CUSTOM_ONLY,
) -> Iterable[Tuple[np.ndarray, List[bool], dict, np.ndarray, int, bool, List[str], bytes]]:
    """
    Create an iterator that replays a bk2 file, yielding frame, keys, annotations, audio samples, actions, and state.

    Args:
        bk2_path (str): Path to the bk2 file to be replayed.
        skip_first_step (bool, optional): Whether to skip the first step of the movie. Defaults to False. For CNeuroMod data, apply to first bk2 of each run.
        game (str, optional): The name of the game. If None, it will be inferred from the bk2 file. Defaults to None.
        scenario (str, optional): The scenario to be used in the emulator. Defaults to None.
        inttype (retro.data.Integrations, optional): The integration type for the emulator. Defaults to retro.data.Integrations.CUSTOM_ONLY.

    Yields:
        tuple: A tuple containing:
            - frame (numpy.ndarray): The current frame of the game.
            - keys (list): The list of keys pressed by the players.
            - annotations (dict): A dictionary containing reward, done, and info.
            - audio_chunk (numpy.ndarray): The PCM audio samples generated for the current step.
            - audio_rate (int): Audio sampling rate in Hz.
            - actions (list): The list of possible actions in the game.
            - state (bytes): The current state of the emulator.
    """
    movie = retro.Movie(bk2_path)
    if game is None:
        game = movie.get_game()
    logging.debug(f"Creating emulator for game: {game}")
    emulator = retro.make(game, state=state, scenario=scenario, inttype=inttype, render_mode=False)
    emulator.initial_state = movie.get_state()
    actions = emulator.buttons
    emulator.reset()
    audio_rate = int(emulator.em.get_audio_rate())
    if skip_first_step:
        movie.step()
    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(emulator.num_buttons):
                keys.append(movie.get_key(i, p))
        frame, rew, terminate, truncate, info = emulator.step(keys)
        annotations = {"reward": rew, "done": terminate, "info": info}
        state = emulator.em.get_state()
        audio_chunk = emulator.em.get_audio().copy()
        yield frame, keys, annotations, audio_chunk, audio_rate, truncate, actions, state
    emulator.close()
    movie.close()

def get_variables_from_replay(
    bk2_fpath,
    skip_first_step=True,
    state=State.DEFAULT,
    game=None,
    scenario=None,
    inttype=retro.data.Integrations.CUSTOM_ONLY,
) -> Tuple[dict, List[dict], List[np.ndarray], List[bytes], np.ndarray, int]:
    """Replay the bk2 file and return game variables and frames."""
    replay = replay_bk2(
        bk2_fpath,
        skip_first_step=skip_first_step,
        state=state,
        game=game,
        scenario=scenario,
        inttype=inttype,
    )
    replay_frames = []
    replay_keys = []
    replay_info = []
    replay_states = []
    annotations = {}
    audio_chunks: List[np.ndarray] = []
    audio_rate = 0

    for frame, keys, annotations, audio_chunk, chunk_rate, _, actions, state in replay:
        replay_keys.append(keys)
        replay_info.append(annotations["info"])
        replay_frames.append(frame)
        replay_states.append(state)
        if audio_chunk.size:
            audio_chunks.append(audio_chunk)
        audio_rate = chunk_rate

    repetition_variables = reformat_info(replay_info, replay_keys, bk2_fpath, actions)

    if not annotations.get("done", False):
        logging.warning(
            f"Done condition not satisfied for {bk2_fpath}. Consider changing skip_first_step."
        )
    audio_track = assemble_audio(audio_chunks)
    return repetition_variables, replay_info, replay_frames, replay_states, audio_track, audio_rate

def reformat_info(info, keys, bk2_fpath, actions):
    """Create a structured dictionary from replay info."""
    filename = op.basename(bk2_fpath)
    entities = filename.split("_")
    entities_dict = {}
    for ent in entities:
        if "-" in ent:
            key, value = ent.split("-", 1)
            entities_dict[key] = value

    repetition_variables = {
        "filename": bk2_fpath,
        "level": entities_dict.get("level"),
        "subject": entities_dict.get("sub"),
        "session": entities_dict.get("ses"),
        "actions": actions,
    }

    for key in info[0].keys():
        repetition_variables[key] = []
    for button in actions:
        repetition_variables[button] = []

    for frame_idx, frame_info in enumerate(info):
        for key in frame_info.keys():
            repetition_variables[key].append(frame_info[key])
        for button_idx, button in enumerate(actions):
            repetition_variables[button].append(keys[frame_idx][button_idx])
    return repetition_variables


def assemble_audio(chunks: List[np.ndarray]) -> np.ndarray:
    """Concatenate audio chunks produced during replay into a single waveform."""

    if not chunks:
        return np.empty(0, dtype=np.int16)
    return np.concatenate(chunks)


def write_wav(audio: np.ndarray, sample_rate: int, output_path: str) -> None:
    """Persist 16-bit PCM audio to disk using the standard library."""

    if audio.size == 0:
        logging.warning("No audio samples provided, skipping WAV write")
        return

    if audio.dtype != np.int16:
        raise ValueError("Audio array must contain int16 samples for WAV export")

    import wave

    channels = 1 if audio.ndim == 1 else audio.shape[1]

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # int16
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())