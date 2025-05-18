
set -e

# run version with binary weighting
uv run topology-learn --algorithm random_sampling --weight-init binary \
    --gym-env-name "CartPole-v1" --n-optim-steps 1000000 --n-init-nodes 5 --n-networks 100 \
    --batch-size 15 --save-folder "data/cartpole_binary" --seed 0 --multi-step \
    --sampling-parameters.connection-prob 0.5 --sampling-parameters.recurrent \
    --sampling-parameters.use-fully-connected-projections --save-buffer-size 10000

uv run topology-learn --algorithm random_sampling --weight-init binary \
    --gym-env-name "ALE/Breakout-v5" --n-optim-steps 1000000 --n-init-nodes 5 --n-networks 100 \
    --batch-size 15 --save-folder "data/breakout_binary" --seed 0 --multi-step \
    --sampling-parameters.connection-prob 0.5 --sampling-parameters.recurrent \
    --sampling-parameters.use-fully-connected-projections --save-buffer-size 10000

uv run topology-learn --algorithm random_sampling --weight-init binary \
    --gym-env-name "ALE/Tetris-v5" --n-optim-steps 1000000 --n-init-nodes 5 --n-networks 100 \
    --batch-size 15 --save-folder "data/tetris_binary" --seed 0 --multi-step \
    --sampling-parameters.connection-prob 0.5 --sampling-parameters.recurrent \
    --sampling-parameters.use-fully-connected-projections --save-buffer-size 10000
