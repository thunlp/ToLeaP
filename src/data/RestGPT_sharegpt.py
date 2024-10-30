import json

def transform_data(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        transformed_data = []
        
        for item in data:
            conversations = []
            
            conversations.append({
                "from": "human",
                "value": item.get("query", "")
            })
            
            # The following operation converts the list in the original "solution" field
            # into a string in the "gpt" field
            solution = item.get("solution", [])
            if isinstance(solution, list):
                solution_str = "\n".join(solution)
            else:
                solution_str = str(solution)

            conversations.append({
                "from": "gpt",
                "value": solution_str
            })
                 
            transformed_item = {
                "conversations": conversations,
            }
            
            transformed_data.append(transformed_item)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, ensure_ascii=False, indent=4)
        
        print(f"Success! Saved to {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    file_pairs = [
        ('RestGPT_data/spotify.json', 'sft_data/spotify_processed.json'),
        ('RestGPT_data/tmdb.json', 'sft_data/tmdb_processed.json') 
    ]
    
    for input_file, output_file in file_pairs:
        transform_data(input_file, output_file)
