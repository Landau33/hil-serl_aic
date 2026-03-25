import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pynput import keyboard

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transistions to collect.")


success_key = False
reset_key = False
active_env = None


def on_press(key):
    global success_key, reset_key, active_env
    try:
        if hasattr(key, "char") and key.char == "h":
            success_key = True
        elif hasattr(key, "char") and key.char == "r":
            reset_key = True
            if active_env is not None and hasattr(active_env.unwrapped, "notify_reset_resume_keypress"):
                active_env.unwrapped.notify_reset_resume_keypress()
    except AttributeError:
        pass


def main(_):
    global success_key, reset_key, active_env
    try:
        from pynput import keyboard
    except ImportError as exc:
        raise RuntimeError("pynput requires a graphical session. Set DISPLAY or run under X11.") from exc

    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
    active_env = env

    obs, _ = env.reset()
    successes = []
    failures = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    
    while len(successes) < success_needed:
        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        obs = next_obs
        if success_key:
            successes.append(transition)
            pbar.update(1)
            success_key = False
        else:
            failures.append(transition)

        if reset_key:
            reset_key = False
            obs, _ = env.reset(options={"wait_for_reset_resume": True})
            reset_key = False
            print("Reset finished. Resuming recording.")

    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(successes, f)
        print(f"saved {success_needed} successful transitions to {file_name}")

    file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(failures, f)
        print(f"saved {len(failures)} failure transitions to {file_name}")
    active_env = None
        
if __name__ == "__main__":
    app.run(main)
