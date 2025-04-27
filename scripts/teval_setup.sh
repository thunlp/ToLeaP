# This file is to set up the T-Eval environment easily.
# Author: Boye Niu
# Date: 2025-04
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

set -e  

echo "Creating T-Eval folder and moving files..."
mkdir -p T-Eval
mv T-Eval_evaluation/* T-Eval/
rm -r T-Eval_evaluation

cd T-Eval

echo "Moving and unzipping teval_data.zip..."
mv ../../data/teval_data.zip ./
unzip teval_data.zip
mv teval_data data

echo "Setup complete!"
echo ""
echo "To evaluate with closed-resource models, run the following:"
echo "bash test_all_teval.sh api <model_name> <display_name> True"
echo ""
echo "Example:"
echo "bash test_all_teval.sh api claude-3-5-sonnet-20240620 claude-3-5-sonnet-20240620 True"

