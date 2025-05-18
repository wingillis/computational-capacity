#!/bin/bash

# run version with binary weighting
set -e

uv run topology-learn --algorithm genetic_evolution --weight-init binary \
    --gym-env-name "CartPole-v1" --n-optim-steps 1000000 --n-init-nodes 5 --n-networks 100 \
    --batch-size 15 --save-folder "data/cartpole_binary_evolution" --seed 0 --multi-step \
    --sampling-parameters.connection-prob 0.5 --sampling-parameters.recurrent \
    --sampling-parameters.use-fully-connected-projections --save-buffer-size 10000 \
    --evolution-parameters.survival-rate 0.5 --evolution-parameters.mutation-rate 0.25 \
    --evolution-parameters.random-sample-rate 0.05

uv run topology-learn --algorithm genetic_evolution --weight-init binary \
    --gym-env-name "ALE/Breakout-v5" --n-optim-steps 1000000 --n-init-nodes 5 --n-networks 100 \
    --batch-size 15 --save-folder "data/breakout_binary_evolution" --seed 0 --multi-step \
    --sampling-parameters.connection-prob 0.5 --sampling-parameters.recurrent \
    --sampling-parameters.use-fully-connected-projections --save-buffer-size 10000 \
    --evolution-parameters.survival-rate 0.5 --evolution-parameters.mutation-rate 0.25 \
    --evolution-parameters.random-sample-rate 0.05