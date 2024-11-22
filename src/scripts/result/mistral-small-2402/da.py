import chardet
import json

file_path = "C:/Users/Owner/Desktop/bohi/BodhiAgent/src/scripts/result/mistral-small-2402/BFCL_v3_simple_result.json"

# 读取原始字节数据
with open(file_path, "rb") as f:
    raw_data = f.read()

# 检测编码
encoding = chardet.detect(raw_data)["encoding"]
print("Detected Encoding:", encoding)

# 用检测出的编码读取
with open(file_path, "r", encoding=encoding) as f:
    data = json.load(f)
print(data)
