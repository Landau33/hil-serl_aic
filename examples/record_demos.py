import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pynput import keyboard

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")


reset_key = False
active_env = None


def on_press(key):
    global reset_key, active_env
    try:
        if hasattr(key, "char") and key.char == "r":
            reset_key = True
            if active_env is not None and hasattr(active_env.unwrapped, "notify_reset_resume_keypress"):
                active_env.unwrapped.notify_reset_resume_keypress()
    except AttributeError:
        pass


def main(_):
    global reset_key, active_env
    try:
        from pynput import keyboard
    except ImportError as exc:
        raise RuntimeError("pynput requires a graphical session. Set DISPLAY or run under X11.") from exc

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    env = None
    pbar = None
    try:
        assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
        config = CONFIG_MAPPING[FLAGS.exp_name]()
        env = config.get_environment(fake_env=False, save_video=False, classifier=True)
        active_env = env
        
        obs, info = env.reset()
        print("Reset done")
        transitions = []
        success_count = 0
        success_needed = FLAGS.successes_needed
        pbar = tqdm(total=success_needed)
        trajectory = []
        returns = 0
        trajectory_succeeded = False
        
        while success_count < success_needed:
            actions = np.zeros(env.action_space.sample().shape) 
            next_obs, rew, done, truncated, info = env.step(actions)
            returns += rew
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
                    infos=info,
                )
            )
            trajectory.append(transition)
            if info.get("succeed", 0):
                trajectory_succeeded = True
            
            pbar.set_description(f"Return: {returns}")
            classifier_prob = info.get("classifier_prob")
            succeed = info.get("succeed", 0)
            # if classifier_prob is not None:
            #     print(
            #         f"classifier_prob={classifier_prob:.4f}, done={done}, truncated={truncated}, succeed={succeed}"
            #     )
            # else:
            #     print(f"done={done}, truncated={truncated}, succeed={succeed}")

            obs = next_obs

            if reset_key:
                reset_key = False
                if trajectory_succeeded:
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    success_count += 1
                    pbar.update(1)
                    print(f"Recorded successful demo #{success_count}. Resetting environment.")
                    if success_count >= success_needed:
                        trajectory = []
                        returns = 0
                        trajectory_succeeded = False
                        break
                else:
                    print("Reset requested before success. Discarding current trajectory.")
                trajectory = []
                returns = 0
                trajectory_succeeded = False
                obs, info = env.reset(options={"wait_for_reset_resume": True})
                reset_key = False
                print("Reset finished. Resuming demo recording.")
                
        if not os.path.exists("./demo_data"):
            os.makedirs("./demo_data")
        uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
        with open(file_name, "wb") as f:
            pkl.dump(transitions, f)
            print(f"saved {success_needed} demos to {file_name}")
    finally:
        active_env = None
        if pbar is not None:
            pbar.close()
        listener.stop()
        if env is not None:
            env.close()

if __name__ == "__main__":
    app.run(main)
