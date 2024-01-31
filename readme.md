Finetune the UNet in Stable Video Diffusion with lora.

Fork from https://github.com/alibaba/animate-anything/

Please modify the configuration file example/train_svd_single_lora.yaml
```python
python train_svd_lora.py --eval # inference
python train_svd_lora.py # train
# or
accelerate launch train_svd_lora.py --eval
accelerate launch train_svd_lora.py
```

