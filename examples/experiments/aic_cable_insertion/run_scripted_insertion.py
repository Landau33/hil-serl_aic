#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np

EXAMPLES_ROOT = Path(__file__).resolve().parents[2]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from experiments.aic_cable_insertion.config import EnvConfig
from experiments.aic_cable_insertion.wrapper import AICCableInsertionEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the scripted cable insertion controller without actor rollout."
    )
    parser.add_argument(
        "--wait-for-reset-resume",
        action="store_true",
        help="Require the usual reset-resume key flow before scripted insertion starts.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Maximum control steps before aborting.",
    )
    parser.add_argument(
        "--status-period",
        type=int,
        default=10,
        help="Print controller status every N steps.",
    )
    return parser.parse_args()


def format_status(step: int, status: dict) -> str:
    return (
        f"step={step} phase={status['phase']} "
        f"lat_err={status['lateral_error_norm']:.4f} "
        f"ax_err={status['axial_error']:+.4f} "
        f"ang_err={status['angular_error_norm']:.4f} "
        f"stuck={status.get('stuck', False)} "
        f"stuck_steps={status.get('stuck_steps', 0)} "
        f"complete={status['complete']}"
    )


def _disable_keyboard_capture(env) -> None:
    """Stop the global pynput listener so terminal keystrokes (e.g. digits in
    the launch command) cannot toggle the scripted intervention off.

    The keyboard wrapper's get_action() still forwards to the scripted
    controller — only the global key event capture is silenced.
    """
    try:
        listener_owner = env._live._intervention
        pynput_listener = listener_owner._listener
    except AttributeError:
        return
    try:
        pynput_listener.stop()
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    env = AICCableInsertionEnv(
        fake_env=False,
        save_video=False,
        config=EnvConfig(),
    )
    zero_action = np.zeros((6,), dtype=np.float32)

    try:
        env.reset(options={"wait_for_reset_resume": args.wait_for_reset_resume})
        _disable_keyboard_capture(env)
        env.start_scripted_intervention()
        print("Scripted insertion started.")

        for step in range(1, args.max_steps + 1):
            _, _, _, truncated, _ = env.step(zero_action)
            status = env.scripted_intervention_status()
            if status is None:
                print("Scripted intervention unavailable.")
                return 1

            if step == 1 or step % args.status_period == 0 or status["complete"]:
                print(format_status(step, status))

            if status["complete"]:
                print("Scripted insertion completed.")
                return 0

            if truncated:
                print("Environment truncated before scripted insertion completed.")
                return 1

        print(f"Scripted insertion did not finish within {args.max_steps} steps.")
        return 1
    finally:
        try:
            env.stop_scripted_intervention()
        except Exception:
            pass
        try:
            time.sleep(0.2)
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
