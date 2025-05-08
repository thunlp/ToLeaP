#!/bin/bash

git clone https://github.com/Junjie-Ye/RoTBench.git

cd RoTBench || { echo "Failed to enter RoTBench directory"; exit 1; }

# Remove all contents except for the data directory.
find . -mindepth 1 -maxdepth 1 ! -name 'Data' -exec rm -rf {} \;

mv Data/* ../../../../data/rotbench

rmdir Data

echo "RoTBench Data Done."

cd ..

rmdir RoTBench
