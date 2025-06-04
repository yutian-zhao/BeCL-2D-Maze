#!/bin/bash

CONFIGS=("--config-path examples/square_maze/contrastive_mi.json --log-dir logs/square_maze --dur 50 --N 1" 
"--config-path examples/square_maze/contrastive_mi.json --log-dir logs/square_maze --dur 50 --N 1" 
"--config-path examples/square_maze/contrastive_mi.json --log-dir logs/square_maze --dur 50 --N 1")

for CONFIG in "${CONFIGS[@]}"
do
  python main.py "$CONFIG" &
done

wait
echo "All runs complete."
