#!/bin/bash

# CONFIGS=("--config-path examples/square_maze/contrastive_mi.json --log-dir logs/square_maze --dur 50 --N 1" 
# "--config-path examples/square_maze/contrastive_mi.json --log-dir logs/square_maze --dur 50 --N 1" 
# "--config-path examples/square_maze/contrastive_mi.json --log-dir logs/square_maze --dur 50 --N 1")

# for CONFIG in "${CONFIGS[@]}"
# do
#   python main.py "$CONFIG" &
# done

# wait
# echo "All runs complete."

#!/bin/bash

# Define the Python script to run
# PYTHON_SCRIPT="/path/to/your_script.py"

# Number of repetitions
REPEAT=3

for i in $(seq 1 $REPEAT); do
    SESSION_NAME="$i"

    echo "Processing tmux session: $SESSION_NAME"

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Session $SESSION_NAME already exists. Sending command to it."
        tmux send-keys -t "$SESSION_NAME" "python main.py --config-path examples/square_maze/endpoint_contrastive_mi_$i.json --log-dir logs/square_maze --dur 50 --N 1" C-m
    else
        echo "Creating new tmux session: $SESSION_NAME"
        tmux new-session -d -s "$SESSION_NAME" "python main.py --config-path examples/square_maze/endpoint_contrastive_mi_$i.json --log-dir logs/square_maze --dur 50 --N 1"
    fi

    sleep 61
done

echo "Done."
