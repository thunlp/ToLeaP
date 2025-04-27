import re
import json

with open('BFCL_results.txt', 'r') as file:
    output = file.read()

pattern = r"🔍 Running test: (\S+)\s+✅ Test completed: \S+\s+🎯 Accuracy: (\d+\.?\d*)"
matches = re.findall(pattern, output)

results = {test: float(acc) for test, acc in matches}

json_output = json.dumps(results, indent=4)
print(json_output)
