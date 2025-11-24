#!/bin/bash

# Caffeinate wrapper to prevent Mac from sleeping during YOLO training
# This keeps your M4 Mac awake during long training sessions

echo "=================================="
echo "Starting YOLO Training with Caffeinate"
echo "=================================="
echo ""
echo "This will prevent your Mac from sleeping during training."
echo "Training output will be shown below."
echo ""
echo "To stop training: Press Ctrl+C"
echo ""
echo "=================================="
echo ""

# Run training with caffeinate
# -d: prevent display from sleeping
# -i: prevent system from idle sleeping
# -m: prevent disk from idle sleeping
caffeinate -dims python scripts/train.py

echo ""
echo "=================================="
echo "Training session ended"
echo "=================================="
