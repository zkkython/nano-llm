from pathlib import Path
def get_config():
    
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "tf_weights",
        "model_basename": "tf_tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tfmodel",
        'dataset_file':'/home/kason/datasets/opus_books/en-it/train-00000-of-00001.parquet',
        'proxy': 'http://127.0.0.1:7890',
    }
    
# 当我们训练挂了，需要从断点继续训练时，需要知道保存的权重文件名   
def get_weights_file_path(config, epoch: str):
    
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = model_basename + str(epoch) + ".pt"
    return str(Path(".")/model_folder/model_filename)