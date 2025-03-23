import json

INPUT_PATH = "metadata_validation.jsonl"
OUTPUT_PATH = "gaia_valid_with_q&a.jsonl"          
MISSING_PATH = "gaia_valid_with_text_tasks.jsonl"
FILTERED_PATH = "gaia_valid_with_text_or_pic_tasks.jsonl"

def filter_fields(input_path, output_path, missing_path, filtered_path):
    total = missing_count = filtered_count = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as out_all, \
         open(missing_path, "w", encoding="utf-8") as out_missing, \
         open(filtered_path, "w", encoding="utf-8") as out_filtered:

        for line in infile:
            total += 1
            obj = json.loads(line)
            record = {
                "Question": obj.get("Question", ""),
                "Final answer": obj.get("Final answer", ""),
                "file_name": obj.get("file_name", "")
            }
            json_line = json.dumps(record, ensure_ascii=False) + "\n"
            out_all.write(json_line)

            # 筛选逻辑
            fname = record["file_name"].lower()
            if fname == "":
                missing_count += 1
                out_missing.write(json_line)
                out_filtered.write(json_line)
                filtered_count += 1
            elif fname.endswith(".jpg") or fname.endswith(".png"):
                out_filtered.write(json_line)
                filtered_count += 1

    print(f"✅ 总记录数: {total}")
    print(f"✅ file_name 为空的记录数: {missing_count}")
    print(f"✅ file_name 为空 或 jpg/png 类型的记录数: {filtered_count}")

if __name__ == "__main__":
    filter_fields(INPUT_PATH, OUTPUT_PATH, MISSING_PATH, FILTERED_PATH)

