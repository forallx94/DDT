# DDT: Decoupled Diffusion Transformer
<div style="text-align: center;">
  <a href="https://arxiv.org/abs/2504.05741"><img src="https://img.shields.io/badge/arXiv-2504.05741-b31b1b.svg" alt="arXiv"></a>
    <a href="https://huggingface.co/spaces/MCG-NJU/DDT"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Online_Demo-green" alt="arXiv"></a>  
</div>

<div style="text-align: center;">
  <a href="https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=ddt-decoupled-diffusion-transformer"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ddt-decoupled-diffusion-transformer/image-generation-on-imagenet-256x256" alt="PWC"></a>
  
<a href="https://paperswithcode.com/sota/image-generation-on-imagenet-512x512?p=ddt-decoupled-diffusion-transformer"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ddt-decoupled-diffusion-transformer/image-generation-on-imagenet-512x512" alt="PWC"></a>
</div>

## Introduction
We decouple diffusion transformer into encoder-decoder design, and surpresingly that a **more substantial encoder yields performance improvements as model size increases**.
![](./figs/main.png)
* We achieves **1.26 FID** on ImageNet256x256 Benchmark with DDT-XL/2(22en6de).
* We achieves **1.28 FID** on ImageNet512x512 Benchmark with DDT-XL/2(22en6de).
* As a byproduct, our DDT can reuse encoder among adjacent steps to accelerate inference.
## Visualizations
![](./figs/teaser.png)
## Checkpoints
We take the off-shelf [VAE](https://huggingface.co/stabilityai/sd-vae-ft-ema) to encode image into latent space, and train the decoder with DDT.

| Dataset     | Model             | Params    | FID  | HuggingFace                                              |
|-------------|-------------------|-----------|------|----------------------------------------------------------|
| ImageNet256 | DDT-XL/2(22en6de) | 675M | 1.26 | [🤗](https://huggingface.co/MCG-NJU/DDT-XL-22en6de-R256) |
| ImageNet512 | DDT-XL/2(22en6de) | 675M | 1.28 | [🤗](https://huggingface.co/MCG-NJU/DDT-XL-22en6de-R512) |
## Online Demos
We provide online demos for DDT-XL/2(22en6de) on HuggingFace Spaces.

HF spases: [https://huggingface.co/spaces/MCG-NJU/DDT](https://huggingface.co/spaces/MCG-NJU/DDT)

## Usages
We use ADM evaluation suite to report FID.
```bash
# for installation
pip install -r requirements.txt
```
```bash
# for inference
python main.py predict -c configs/repa_improved_ddt_xlen22de6_256.yaml --ckpt_path=XXX.ckpt
```

```bash
# for training
# extract image latent (optional)
python3 tools/cache_imlatent4.py
# train
python main.py fit -c configs/repa_improved_ddt_xlen22de6_256.yaml
```


## Reference
```bibtex
@ARTICLE{ddt,
  title         = "DDT: Decoupled Diffusion Transformer",
  author        = "Wang, Shuai and Tian, Zhi and Huang, Weilin and Wang, Limin",
  month         =  apr,
  year          =  2025,
  archivePrefix = "arXiv",
  primaryClass  = "cs.CV",
  eprint        = "2504.05741"
}
```