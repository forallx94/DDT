# DDT: Decoupled Diffusion Transformer
## Introduction
We decouple diffusion transformer into encoder-decoder design, and surpresingly that a **more substantial encoder yields performance improvements as model size increases**.
![](./figs/main.png)
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