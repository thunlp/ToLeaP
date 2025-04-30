cd ../../data
mkdir TaskBench
wget https://github.com/microsoft/JARVIS/blob/main/taskbench/data_dailylifeapis/tool_desc.json
mv tool_desc.json tool_desc_dailylifeapis.json
wget https://github.com/microsoft/JARVIS/blob/main/taskbench/data_huggingface/tool_desc.json
mv tool_desc.json tool_desc_huggingface.json
wget https://github.com/microsoft/JARVIS/blob/main/taskbench/data_multimedia/tool_desc.json
mv tool_desc.json tool_desc_multimedia.json