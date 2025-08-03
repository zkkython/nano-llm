import os
from transformer_train import train_model       
from config import get_config
if __name__ == "__main__":  
            
    config = get_config()
     # 看具体情况，看是否需要设置代理
    # 设置 HTTP/HTTPS 代理（根据你的代理地址修改）
    if config['proxy']:
        os.environ["http_proxy"] = "http://127.0.0.1:7890"
        os.environ["https_proxy"] = "http://127.0.0.1:7890"
    train_model(config)