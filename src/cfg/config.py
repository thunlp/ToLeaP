# This file is ...
# Author: Haotian Chen, ...
# Date: 2024-08
# Copyright (c) THUNLP, Tsinghua University. All rights reserved. 
# See LICENSE file in the project root for license information. 

import os
from dotenv import load_dotenv

# loading variables stored in .env
load_dotenv()

class Config:
    api_key = os.getenv("API_KEY")
    api_model = os.getenv("API_MODEL")
    api_base = os.getenv("DATABASE_URL")
    data_folder_path = os.getenv("DATA_FOLDER_PATH")

    port = os.getenv("PORT")
    host = os.getenv("HOST")
    use_hf = os.getenv("USE_HF")

    use_llama = os.getenv("USE_LLAMA") == "True"
    llama_model_path = os.getenv("LLAMA_MODEL_PATH")
    
    max_past_message_include = 128000
    default_system_prompt = ''

    @staticmethod
    def display_config():
        print(f"API Key: {Config.api_key}")
        print(f"Use Llama: {Config.use_llama}")
        print(f"Llama Model Path: {Config.llama_model_path}")


if __name__ == "__main__":
    Config.display_config()