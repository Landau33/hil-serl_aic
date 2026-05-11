#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import orbax.checkpoint as ocp
import os
import copy
import pickle as pkl
from typing import TYPE_CHECKING
from gymnasium.wrappers.common import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING

if TYPE_CHECKING:
    from pynput import keyboard

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def dump_pickle_atomic(obj, path):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pkl.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_transitions_from_dir(dir_path, target_buffer, buffer_name):
    loaded_files = 0
    skipped_files = 0

    for file in natsorted(glob.glob(os.path.join(dir_path, "*.pkl"))):
        if os.path.getsize(file) == 0:
            print(
                f"Warning: skipping empty {buffer_name} file '{file}'. "
                "It was likely left behind by an interrupted write."
            )
            skipped_files += 1
            continue

        try:
            with open(file, "rb") as f:
                transitions = pkl.load(f)
        except (EOFError, pkl.UnpicklingError) as exc:
            print(f"Warning: skipping unreadable {buffer_name} file '{file}': {exc}")
            skipped_files += 1
            continue

        for transition in transitions:
            target_buffer.insert(transition)
        loaded_files += 1

    return loaded_files, skipped_files


def _checkpoint_step_from_path(checkpoint_path: str) -> str:
    return os.path.basename(os.path.normpath(checkpoint_path)).removeprefix(
        "checkpoint_"
    )


def _resolve_checkpoint_path(
    checkpoint_root: str, step: int | None = None
) -> str | None:
    checkpoint_root = os.path.abspath(checkpoint_root)
    if step is not None:
        candidate = os.path.join(checkpoint_root, f"checkpoint_{step}")
        if os.path.exists(candidate):
            return candidate
    return checkpoints.latest_checkpoint(checkpoint_root)


def _restore_agent_state(checkpoint_root: str, target_state, step: int | None = None):
    checkpoint_path = _resolve_checkpoint_path(checkpoint_root, step=step)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_root}")
    if os.path.isdir(checkpoint_path):
        restored_state = checkpoints.restore_checkpoint(
            checkpoint_path,
            target_state,
            orbax_checkpointer=ocp.PyTreeCheckpointer(),
        )
    else:
        restored_state = checkpoints.restore_checkpoint(
            checkpoint_path,
            target_state,
        )
    return restored_state, checkpoint_path


active_env = None
reset_key = False
success_key = False


def on_press(key):
    global active_env, reset_key, success_key
    try:
        if hasattr(key, "char") and key.char == "r":
            reset_key = True
            if active_env is not None and hasattr(
                active_env.unwrapped, "notify_reset_resume_keypress"
            ):
                active_env.unwrapped.notify_reset_resume_keypress()
        elif hasattr(key, "char") and key.char == "h":
            success_key = True
    except AttributeError:
        pass


##############################################################################


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    global active_env, reset_key, success_key
    try:
        from pynput import keyboard
    except ImportError as exc:
        raise RuntimeError(
            "pynput requires a graphical session. Set DISPLAY or run under X11."
        ) from exc

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    active_env = env
    try:
        if FLAGS.eval_checkpoint_step:
            success_counter = 0
            time_list = []

            ckpt, checkpoint_path = _restore_agent_state(
                FLAGS.checkpoint_path,
                agent.state,
                step=FLAGS.eval_checkpoint_step,
            )
            agent = agent.replace(state=ckpt)
            print_green(f"Loaded evaluation checkpoint from {checkpoint_path}.")

            for episode in range(FLAGS.eval_n_trajs):
                obs, _ = env.reset(options={"wait_for_reset_resume": True})
                print("Reset finished. Resuming actor rollout.")
                done = False
                start_time = time.time()
                while not done:
                    sampling_rng, key = jax.random.split(sampling_rng)
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs), argmax=False, seed=key
                    )
                    actions = np.asarray(jax.device_get(actions))

                    next_obs, reward, done, truncated, info = env.step(actions)
                    obs = next_obs

                    if done:
                        if reward:
                            dt = time.time() - start_time
                            time_list.append(dt)
                            print(dt)

                        success_counter += reward
                        print(reward)
                        print(f"{success_counter}/{episode + 1}")
                        print("Episode finished. Press 'r' to begin reset.")
                        while not reset_key:
                            time.sleep(0.05)
                        reset_key = False

            print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
            print(f"average time: {np.mean(time_list)}")
            return  # after done eval, return and exit

        start_step = (
            int(
                os.path.basename(
                    natsorted(
                        glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))
                    )[-1]
                )[12:-4]
            )
            + 1
            if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
            else 0
        )

        datastore_dict = {
            "actor_env": data_store,
            "actor_env_intvn": intvn_data_store,
        }

        client = TrainerClient(
            "actor_env",
            FLAGS.ip,
            make_trainer_config(),
            data_stores=datastore_dict,
            wait_for_server=True,
            timeout_ms=3000,
        )

        # Function to update the agent with new params
        def update_params(params):
            nonlocal agent
            agent = agent.replace(state=agent.state.replace(params=params))

        client.recv_network_callback(update_params)

        transitions = []
        demo_transitions = []

        obs, _ = env.reset(options={"wait_for_reset_resume": True})
        print("Reset finished. Resuming actor rollout.")
        done = False

        # training loop
        timer = Timer()
        running_return = 0.0
        episode_step_count = 0
        already_intervened = False
        success_streak = 0
        pending_success_transitions = []
        intervention_count = 0
        intervention_steps = 0
        pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
        for step in pbar:
            timer.tick("total")

            if reset_key:
                reset_key = False
                success_key = False
                success_streak = 0
                pending_success_transitions = []
                running_return = 0.0
                episode_step_count = 0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                print("Reset requested. Waiting for reset resume key 'r'.")
                obs, _ = env.reset(options={"wait_for_reset_resume": True})
                reset_key = False
                done = False
                truncated = False
                print("Reset finished. Resuming actor rollout.")

            with timer.context("sample_actions"):
                if step < config.random_steps:
                    actions = env.action_space.sample()
                else:
                    sampling_rng, key = jax.random.split(sampling_rng)
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        seed=key,
                        argmax=False,
                    )
                    actions = np.asarray(jax.device_get(actions))

            # Step environment
            with timer.context("step_env"):

                next_obs, reward, env_done, truncated, info = env.step(actions)
                episode_step_count += 1
                if "left" in info:
                    info.pop("left")
                if "right" in info:
                    info.pop("right")

                frame_succeed = bool(info.get("succeed", 0))
                success_streak = success_streak + 1 if frame_succeed else 0
                manual_success = success_key
                success_key = False
                streak_threshold_reached = success_streak >= 2
                success_threshold_reached = streak_threshold_reached or manual_success
                done = bool(truncated or success_threshold_reached)
                if done:
                    if success_threshold_reached:
                        reward = 1
                        info["succeed"] = 1
                    info["done_source"] = (
                        "manual_success"
                        if manual_success
                        else (
                            "success_streak"
                            if streak_threshold_reached
                            else (
                                "truncated"
                                if truncated
                                else "env_done" if env_done else "unknown"
                            )
                        )
                    )

                # override the action with the intervention action
                if "intervene_action" in info:
                    actions = info.pop("intervene_action")
                    intervention_steps += 1
                    if not already_intervened:
                        intervention_count += 1
                        print(f"Manual intervention started: action={actions}")
                    already_intervened = True
                else:
                    if already_intervened:
                        print("Manual intervention released. Policy control resumed.")
                    already_intervened = False

                transition = dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done,
                )
                if "grasp_penalty" in info:
                    transition["grasp_penalty"] = info["grasp_penalty"]

                def commit_transition(committed_transition, intervened):
                    data_store.insert(committed_transition)
                    transitions.append(copy.deepcopy(committed_transition))
                    if intervened:
                        intvn_data_store.insert(committed_transition)
                        demo_transitions.append(copy.deepcopy(committed_transition))
                    return float(committed_transition["rewards"])

                if success_threshold_reached:
                    for (
                        pending_transition,
                        pending_intervened,
                    ) in pending_success_transitions[-2:]:
                        pending_transition["rewards"] = 1
                        running_return += commit_transition(
                            pending_transition, pending_intervened
                        )
                    pending_success_transitions = []
                    running_return += commit_transition(transition, already_intervened)
                elif frame_succeed and not done:
                    transition["rewards"] = 0
                    pending_success_transitions.append(
                        (copy.deepcopy(transition), already_intervened)
                    )
                else:
                    for (
                        pending_transition,
                        pending_intervened,
                    ) in pending_success_transitions:
                        pending_transition["rewards"] = 0
                        running_return += commit_transition(
                            pending_transition, pending_intervened
                        )
                    pending_success_transitions = []
                    running_return += commit_transition(transition, already_intervened)

                obs = next_obs
                if done or truncated:
                    lat_err = info.get("reward_lateral_error_m")
                    ax_err = info.get("reward_axial_error_m")
                    ang_err = info.get("reward_angle_error_rad")
                    lat_str = (
                        f"{lat_err:.4f}m"
                        if isinstance(lat_err, (int, float))
                        else "N/A"
                    )
                    ax_str = (
                        f"{ax_err:.4f}m" if isinstance(ax_err, (int, float)) else "N/A"
                    )
                    ang_str = (
                        f"{ang_err:.4f}rad"
                        if isinstance(ang_err, (int, float))
                        else "N/A"
                    )
                    print(
                        "Episode end:"
                        f" done={done}, truncated={truncated},"
                        f" reward={reward},"
                        f" episode_return={running_return},"
                        f" succeed={info.get('succeed', 'N/A')},"
                        f" lat_err={lat_str},"
                        f" ax_err={ax_str},"
                        f" ang_err={ang_str},"
                        f" reward_streak={info.get('reward_met_streak', 'N/A')},"
                        f" success_streak={success_streak},"
                        f" done_source={info.get('done_source', 'N/A')}"
                    )
                    info.setdefault(
                        "episode",
                        {
                            "r": running_return,
                            "l": episode_step_count,
                            "t": 0.0,
                        },
                    )
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps
                    stats = {"environment": info}  # send stats to the learner to log
                    client.request("send-stats", stats)
                    client.update()
                    pbar.set_description(f"last return: {running_return}")
                    print("Episode finished. Resetting environment automatically.")
                    success_streak = 0
                    pending_success_transitions = []
                    running_return = 0.0
                    episode_step_count = 0
                    intervention_count = 0
                    intervention_steps = 0
                    already_intervened = False
                    obs, _ = env.reset(options={"wait_for_reset_resume": True})
                    reset_key = False
                    print("Reset finished. Resuming actor rollout.")

            if (
                step > 0
                and config.buffer_period > 0
                and step % config.buffer_period == 0
            ):
                # dump to pickle file
                buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
                demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
                if not os.path.exists(buffer_path):
                    os.makedirs(buffer_path)
                if not os.path.exists(demo_buffer_path):
                    os.makedirs(demo_buffer_path)
                dump_pickle_atomic(
                    transitions,
                    os.path.join(buffer_path, f"transitions_{step}.pkl"),
                )
                transitions = []
                dump_pickle_atomic(
                    demo_transitions,
                    os.path.join(demo_buffer_path, f"transitions_{step}.pkl"),
                )
                demo_transitions = []

            timer.tock("total")

            if step % config.log_period == 0:
                stats = {"timer": timer.get_average_times()}
                client.request("send-stats", stats)
    finally:
        active_env = None
        listener.stop()


##############################################################################


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = (
        int(
            os.path.basename(
                checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
            )[11:]
        )
        + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset(
            {"critic", "grasp_critic", "actor", "temperature"}
        )

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)

    if (
        config.setup_mode == "single-arm-fixed-gripper"
        or config.setup_mode == "dual-arm-fixed-gripper"
    ):
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == "single-arm-learned-gripper":
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == "dual-arm-learned-gripper":
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt, checkpoint_path = _restore_agent_state(
            FLAGS.checkpoint_path,
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        ckpt_number = _checkpoint_step_from_path(checkpoint_path)
        print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if "infos" in transition and "grasp_penalty" in transition["infos"]:
                        transition["grasp_penalty"] = transition["infos"][
                            "grasp_penalty"
                        ]
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            loaded_files, skipped_files = load_transitions_from_dir(
                os.path.join(FLAGS.checkpoint_path, "buffer"),
                replay_buffer,
                "replay buffer",
            )
            print_green(
                "Loaded previous buffer data. "
                f"Replay buffer size: {len(replay_buffer)} "
                f"(loaded {loaded_files} files, skipped {skipped_files})."
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            loaded_files, skipped_files = load_transitions_from_dir(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer"),
                demo_buffer,
                "demo buffer",
            )
            print_green(
                "Loaded previous demo buffer data. "
                f"Demo buffer size: {len(demo_buffer)} "
                f"(loaded {loaded_files} files, skipped {skipped_files})."
            )

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
