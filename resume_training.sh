#!/bin/bash

# Caffeinate wrapper to resume YOLO training
# This keeps your Mac awake during training continuation

echo "=================================="
echo "Resuming YOLO Training with Caffeinate"
echo "=================================="
echo ""
echo "This will prevent your Mac from sleeping during training."
echo "Training output will be shown below."
echo ""
echo "To stop training: Press Ctrl+C"
echo ""
echo "=================================="
echo ""

# Resume training with caffeinate
caffeinate -dims python scripts/resume_training.py

echo ""
echo "=================================="
echo "Training session ended"
echo "=================================="
