# DreamRenderer: Taming Multi-Instance Attribute Control in Large-Scale Text-to-Image Models 🎨

[[Project Page]](https://limuloo.github.io/DreamRenderer/) [[Paper]](https://arxiv.org/abs/2503.12885) [[Hugging Face]](https://huggingface.co/papers/2503.12885) [[Supplementary Material]](https://drive.google.com/file/d/1MNaKZmIyBXT7Ia_6DJ56vJ2TeB5o8m6c/view?usp=sharing)

## 🔥 News

- 2025-03-17: Our paper [DreamRenderer](https://arxiv.org/abs/2503.12885) is now available on arXiv and [Supplementary Material](https://drive.google.com/file/d/1MNaKZmIyBXT7Ia_6DJ56vJ2TeB5o8m6c/view?usp=sharing) is released.

![Multi-Instance Attribute Control](static/images/teaser.png)

**Codes will be released this week, stay tuned! 🚀**

## 📝 Introduction

DreamRenderer is a training-free method built upon the FLUX model that enables users to precisely control the content of each instance through bounding boxes or masks while ensuring overall visual harmony. 

## ✅ To-Do List

- [x] Arxiv Paper & Supplementary Material
- [ ] More Demos
- [ ] Inference Code 
- [ ] ComfyUI support
- [ ] Huggingface Space support

## 🛠️ Installation

### 🚀 Checkpoints

Download the checkpoint of the SAM2, [sam2_hiera_large.pt](https://drive.google.com/file/d/1QjdY64w7pKm8smh0bV7K9-joeZiow8e0/view?usp=sharing).


### 💻 Environment Setup

```bash
conda create -n dreamrenderer python=3.10 -y
conda activate dreamrenderer
cd segment-anything-2
pip install -e . --no-deps
cd ..
pip install -r requirements.txt
```

## 📊 Comparison with Other Models

<p align="center">
  <img src="static/images/rerendering.png" alt="comparison"/>
</p>

## 📚 Citation

If you find this repository useful, please cite using the following BibTeX entry:

```bibtex
@misc{zhou2025dreamrenderer,
      title={DreamRenderer: Taming Multi-Instance Attribute Control in Large-Scale Text-to-Image Models},
      author={Dewei Zhou and Mingwei Li and Zongxin Yang and Yi Yang},
      year={2025},
      eprint={2503.12885},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.12885},
}
```

## 📬 Contact

If you have any questions or suggestions, please feel free to contact us 😆!
