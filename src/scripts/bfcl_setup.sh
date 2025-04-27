# This file is to set up the bfcl environment easily.
# Author: Boye Niu
# Date: 2025-04
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

set -e

# Clone the Gorilla repository
echo "Cloning Gorilla repository..."
git clone https://github.com/ShishirPatil/gorilla.git

# Change directory to `berkeley-function-call-leaderboard`
echo "Navigating to berkeley-function-call-leaderboard..."
cd gorilla/berkeley-function-call-leaderboard

# Install the package in editable mode
echo "Installing berkeley-function-call-leaderboard package in editable mode..."
pip install -e .

# Install additional dependencies for oss_eval_vllm
echo "Installing oss_eval_vllm dependencies..."
pip install -e .[oss_eval_vllm]

echo "BFCL setup completed successfully!"
