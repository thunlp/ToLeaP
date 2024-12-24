
set -e


echo "Cloning T-Eval repository..."
git clone https://github.com/open-compass/T-Eval.git


cd T-Eval


echo "Installing T-Eval dependencies..."
pip install -r requirements.txt
pip install vllm==0.6.1

echo "Cloning lagent repository..."
git clone https://github.com/InternLM/lagent.git


cd lagent
echo "Installing lagent package..."
pip install -e .

cd ..

if [ -d "../teval_data" ]; then
    echo "Moving 'data' directory to T-Eval..."
    mv ../teval_data ./data
else
    echo "'data' directory not found. Skipping move step."
fi

echo "Setup completed successfully!"
