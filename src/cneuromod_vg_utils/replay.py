import retro
import logging
from retro.enums import State
import os.path as op

def replay_bk2(
    bk2_path, skip_first_step=False, state=State.DEFAULT, game=None, scenario=None, inttype=retro.data.Integrations.CUSTOM_ONLY
):
    """
    Create an iterator that replays a bk2 file, yielding frame, keys, annotations, sound, actions, and state.

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
            - sound (None): Placeholder for sound data (currently not used).
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
        yield frame, keys, annotations, truncate, actions, state
    emulator.close()
    movie.close()

def get_variables_from_replay(
    bk2_fpath,
    skip_first_step=True,
    state=State.DEFAULT,
    game=None,
    scenario=None,
    inttype=retro.data.Integrations.CUSTOM_ONLY,
):
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

    for frame, keys, annotations, _, actions, state in replay:
        replay_keys.append(keys)
        replay_info.append(annotations["info"])
        replay_frames.append(frame)
        replay_states.append(state)

    repetition_variables = reformat_info(replay_info, replay_keys, bk2_fpath, actions)

    if not annotations.get("done", False):
        logging.warning(
            f"Done condition not satisfied for {bk2_fpath}. Consider changing skip_first_step."
        )
    return repetition_variables, replay_info, replay_frames, replay_states

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