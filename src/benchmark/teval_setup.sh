
set -e


echo "Cloning T-Eval repository..."
git clone https://github.com/open-compass/T-Eval.git


cd T-Eval


echo "Installing T-Eval dependencies..."
pip install -r requirements.txt


echo "Cloning lagent repository..."
git clone https://github.com/InternLM/lagent.git


cd lagent
echo "Installing lagent package..."
pip install -e .

cd ..

if [ -d "../data" ]; then
    echo "Moving 'data' directory to T-Eval..."
    mv ../data ./data
else
    echo "'data' directory not found. Skipping move step."
fi

echo "Setup completed successfully!"
