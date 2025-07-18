# Unconditional DDPM for Armenian Letters

This project implements an **unconditional Denoising Diffusion Probabilistic Model (DDPM)** trained on a custom dataset of **19,608 images of Armenian letters** in approximately **258 fonts**. The final model is capable of generating high-quality images of Armenian letters in diverse typographic styles — and sometimes hallucinates glyphs with similar structural features, even if they do not correspond to any real Armenian letter.

## 👀 Overview

This project explores generative modeling of Armenian typography using **unconditional DDPMs**. The core components of the system include:

- A UNet-based DDPM model that learns to denoise images through a reverse diffusion process.
- A fast and effective inference module using **DPM-Solver** (used with 20 steps), which drastically reduces sampling time compared to naive DDPM sampling.
- A dataset of grayscale 96x96 letter images generated from Armenian Unicode characters in diverse font styles.

The goal is to enable generation of letterforms that reflect the stylistic richness of Armenian typography. The model is capable of:

- Reconstructing stylistically varied Armenian characters.
- Generating novel glyphs that are structurally similar to existing letters but may not correspond to real characters.

## 📜 Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Training](#training)
- [Results](#results)
- [Notes](#notes)
- [References](#references)

## 🗂️ Project Structure

```
ddpm/
├── fonts/
│   ├── lower_ա_1.png
│   ├── lower_ա_2.png
│   └── ...
├── outputs/
│   ├── checkpoints/
│   ├── samples/
│   ├── test/
│   └── loss/
├── input/
├── input-test/
├── utils/
│   ├── utils.py
│   └── prepare-dataset.py
├── config.py
├── models.py
├── dataset.py
└── train.py
````

## 🧾 Prerequisites

- PyTorch
- Pillow (PIL)
- Matplotlib
- NumPy

## 🚀 Training

Train the DDPM model using:

```bash
python train.py
```
Training parameters can be modified in the `config.py` file.

The training process is designed to save key outputs for monitoring and evaluation:  

- **Loss Tracking**: A figure showing the evolution of diffusion loss is saved in the `loss` folder in `outputs`.  
- **Sample Generation**: Intermediate generated samples are saved in the `samples` folder in `outputs` to track training progress.  
- **Model Checkpoints**: Weights are periodically saved in the `checkpoints` folder in `outputs`, with support for checkpoint loading.  
- **Data Evaluation**: During training, the model generates and saves samples in the `test` folder in `outputs` to visualize performance.  

## 📈 Results
The results are satisfactory; however, the model would benefit from further training. The results are shown for around 300k steps.

![training-results](/images/training-sample.png "Training Results")
![training-results-2](/images/training-sample-2.png "Training Results")

## 🧾 Notes

* The model is **unconditional**, meaning the generation is not guided with content or style, i.e. no conditions are provided for generation.
* Occasionally generates letter-like structures that do not correspond to real Armenian letters — a natural effect of training on a narrow but stylistically rich distribution.

As shown in the following images, some generated letters have completely different strcuture, while others were slightly changed:

![new-letters](/images/new-letters.png "Training Results")
![new-letters-2](/images/new-letters-2.png "Training Results")

## 📚 References

* [Ho et al., 2020. Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
* [Lu et al., 2022. DPM-Solver: Fast ODE Solvers for Diffusion Models](https://arxiv.org/abs/2206.00927)