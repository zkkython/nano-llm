from textwrap import indent
from transformers import AutoModelForCausalLM
import modelscope
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
import json
import torch
model_path = modelscope.snapshot_download("Qwen/Qwen3-0.6B", local_dir="./") # 下载模型
# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

model_class = model.model 
print(type(model_class))
print(str(model_class))
print((model.config))

def parse_model_structure(model):
    structure = {}
    for name, module in model.named_modules():
        print(f'{name} : {module}')
        # if isinstance(module, torch.nn.Module):
        #     module_info = {
        #         "type": module.__class__.__name__,
        #         "params": {k: getattr(module, k) for k in ["in_features", "out_features", "num_embeddings", "embedding_dim"] 
        #                   if hasattr(module, k)},
        #         "shape": dict(module.weight.shape) if hasattr(module, "weight") else None
        #     }
        #     structure[name] = module_info
    return structure

model_structure = parse_model_structure(model)
print(json.dumps(model_structure, indent=2))


print('+++++++++++++++')
def auto_parse(model):
    # 综合配置和模块信息
    info = {
        "config": model.config.to_dict(),
        "modules": {}
    }
    
    # 遍历所有子模块
    for name, module in model.named_modules():
        if name == "":  # 跳过根模块
            continue
        module_info = {
            "class": module.__class__.__name__,
            "parameters": {k: v.shape for k, v in module.named_parameters()}
        }
        info["modules"][name] = module_info
    
    return info

print(json.dumps(auto_parse(model), indent=2))
