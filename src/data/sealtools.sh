#dowload sealtools
mkdir -p sft_data
mkdir -p sft_data/sealtools
git clone https://github.com/fairyshine/Seal-Tools.git

SOURCE_DIR="Seal-Tools/Seal-Tools_Dataset/dataset_for_finetune"
DEST_DIR="sft_data/sealtools"

echo "Copying Sealtools datasetss..."
mv "$SOURCE_DIR"/* "$DEST_DIR"

rm -rf Seal-Tools
