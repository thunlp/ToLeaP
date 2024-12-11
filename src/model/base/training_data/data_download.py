from datasets import load_dataset

ds = load_dataset("Salesforce/xlam-function-calling-60k")  # 
ds = load_dataset("Locutusque/function-calling-chatml")  # 
ds = load_dataset("glaiveai/glaive-code-assistant-v3")  # 
ds = load_dataset("Team-ACE/ToolACE")  # 
ds = load_dataset("llamafactory/glaive_toolcall_en")  # 
ds = load_dataset("argilla-warehouse/python-lib-tools-v0.1")  # 
ds = load_dataset("llamafactory/glaive_toolcall_zh")  # 2.38M
ds = load_dataset("reasonwang/ToolGen-Datasets", "toolgen_atomic")  # 1.63G
ds = load_dataset("reasonwang/ToolGen-Datasets", "toolgen_atomic_memorization")  # 31.3M
ds = load_dataset("reasonwang/ToolGen-Datasets", "toolgen_atomic_retrieval")  # 255M

# 4.11G + 5.73G + 268M + 214M + 159k+562k+363k + 20.1M + 6.60M + 3.96G + 188M + 5.31G + 671k+427k + 1.27G + 1.21G + 
# ds = load_dataset("ghh001/InstructIE_tool")  # datasets.exceptions.DatasetGenerationError

ds = load_dataset("argilla-warehouse/python-seed-tools")  # 7.99M
ds = load_dataset("danilopeixoto/pandora-tool-calling")  # 84.5M
ds = load_dataset("huggingface-tools/default-prompts")  # 

ds = load_dataset("roborovski/synthetic-tool-calls-v2-dpo-pairs")  # 2.10M
ds = load_dataset("Yhyu13/ToolBench_toolllama_G123_dfs")  # 2.00G
ds = load_dataset("minyichen/glaive_toolcall_zh_tw")
ds = load_dataset("BitAgent/tool_calling")
ds = load_dataset("jvhoffbauer/gsm8k-toolcalls")
ds = load_dataset("interstellarninja/tool-calls-sampled-prompts")
ds = load_dataset("interstellarninja/tool-calls-multiturn")
ds = load_dataset("interstellarninja/tool-calls-singleturn")

# instruction finetuning data
# ds = load_dataset("microsoft/orca-agentinstruct-1M-v1")
# ds = load_dataset("O1-OPEN/OpenO1-SFT")
# ds = load_dataset("amphora/QwQ-LongCoT-130K")
# ds = load_dataset("HuggingFaceTB/smoltalk")


