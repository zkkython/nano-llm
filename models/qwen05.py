from transformers import AutoModelForCausalLM
import modelscope
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
model_path = modelscope.snapshot_download("Qwen/Qwen3-0.6B", local_dir="./") # 下载模型
# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

model_class = model.model 
print(type(model_class))
print(str(model_class))
print((model.config))