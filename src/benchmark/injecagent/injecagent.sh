#!/bin/bash

git clone https://github.com/uiuc-kang-lab/InjecAgent.git

cd InjecAgent || { echo "Failed to enter InjecAgent directory"; exit 1; }

# Remove all contents except for the data directory.
find . -mindepth 1 -maxdepth 1 ! -name 'data' -exec rm -rf {} \;

mv data/* ../../../../data/injecagent

rmdir data

echo "InjecAgent Data Done."

cd ..

rmdir InjecAgent
