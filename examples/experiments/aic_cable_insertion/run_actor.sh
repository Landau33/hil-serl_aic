export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=aic_cable_insertion \
    --checkpoint_path=./checkpoints_test2 \
    --actor
