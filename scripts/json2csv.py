import json
import csv
import ast
from json import JSONDecodeError

def parse_line(line):
    """
    先尝试 json.loads，失败时用 ast.literal_eval。
    """
    try:
        return json.loads(line)
    except JSONDecodeError:
        # 降级处理：把 Python 字面量转成 dict
        return ast.literal_eval(line)

def process_json_line(j):
    """
    判断 JSON 对象的结构：
    - 如果所有顶层值均为 dict，则返回 (header, rows)
      header 的第一列为 "Category"，后续列按 sample.keys() 顺序；
    - 否则，将 j 当做平铺 dict 处理，header 为排序后的所有键，rows 只有一行。
    """
    if all(isinstance(v, dict) for v in j.values()):
        sample = next(iter(j.values()))
        headers = ["Category"] + list(sample.keys())
        rows = []
        for category, subdict in j.items():
            row = [category] + [subdict.get(k, "") for k in sample.keys()]
            rows.append(row)
    else:
        # 平铺字典：对键排序，保证列顺序一致
        headers = sorted(j.keys())
        rows = [[j.get(k, "") for k in headers]]
    return headers, rows

def process_json_file(json_file, csv_file):
    with open(json_file, 'r', encoding='utf-8') as f, \
         open(csv_file, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = parse_line(line)
                header, rows = process_json_line(j)
                writer.writerow(header)
                writer.writerows(rows)
                writer.writerow([])  # 空行分隔
            except Exception as e:
                print(f"处理出错：{line}\n错误信息：{e}")

if __name__ == '__main__':
    process_json_file('Qwen3-8B_results.json', 'Qwen3-8B_results.csv')
    print("转换完成。")
