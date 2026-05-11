export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
INFERENCE_CKPT_PATH="$(python3 -c 'from pathlib import Path; import importlib.util; p=Path("/home/yuang/ws_aic/aic/aic_example_policies/aic_example_policies/ros/hil_serl/config.py"); spec=importlib.util.spec_from_file_location("aic_hilserl_config", p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print(m.HilSerlModelConfig().checkpoint_path)')" && \
INFERENCE_CKPT_STEP="$(python3 -c 'from pathlib import Path; import importlib.util; p=Path("/home/yuang/ws_aic/aic/aic_example_policies/aic_example_policies/ros/hil_serl/config.py"); spec=importlib.util.spec_from_file_location("aic_hilserl_config", p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print(m.HilSerlModelConfig().checkpoint_step)')" && \
python ../../train_rlpd.py "$@" \
    --exp_name=aic_cable_insertion \
    --checkpoint_path="${INFERENCE_CKPT_PATH}" \
    --eval_checkpoint_step="${INFERENCE_CKPT_STEP}" \
    --eval_n_trajs="${EVAL_N_TRAJS:-10}" \
    --debug \
    --actor
