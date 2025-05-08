# This file is to transform the BFCL rsults into clean format.
# Author: Zijun Song
# Date: 2025-04
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

import re
import json

with open('BFCL_results.txt', 'r') as file:
    output = file.read()

pattern = r"🔍 Running test: (\S+)\s+✅ Test completed: \S+\s+🎯 Accuracy: (\d+\.?\d*)"
matches = re.findall(pattern, output)

results = {test: float(acc) for test, acc in matches}

json_output = json.dumps(results, indent=4)
print(json_output)
