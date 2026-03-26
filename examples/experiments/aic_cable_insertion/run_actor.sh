GPU_UUID="${GPU_UUID:-GPU-45a924c1-5d32-a7cc-ed8d-674d0179dcee}" && \
export CUDA_VISIBLE_DEVICES="${GPU_UUID}" && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=aic_cable_insertion \
    --checkpoint_path=./checkpoints_test \
    --debug \
    --actor
