# DDT: Decoupled Diffusion Transformer
## Introduction
We decouple diffusion transformer into encoder-decoder design, and surpresingly that a **more substantial encoder yields performance improvements as model size increases**.
![](./figs/main.png)
* We achieves **1.26 FID** on ImageNet256x256 Benchmark with DDT-XL/2(22en6de).
* We achieves **1.28 FID** on ImageNet512x512 Benchmark with DDT-XL/2(22en6de).
* As a byproduct, our DDT can reuse encoder among adjacent steps to accelerate inference.
## Visualizations
![](./figs/teaser.png)
## Usgae
```bash
# for training
python main.py fit -c configs/repa_improved_ddt_xlen22de6_256.yaml
```

```bash
# for inference
python main.py predict -c configs/repa_improved_ddt_xlen22de6_256.yaml --ckpt_path=XXX.ckpt
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