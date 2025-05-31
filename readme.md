# NVIDIA Data Parallelism: How to Train Deep Learning Models on Multiple GPUs

This repository contains the complete materials for the course **"Data Parallelism: How to Train Deep Learning Models on Multiple GPUs"**. The course is designed to help machine learning practitioners understand and implement efficient multi-GPU training using **PyTorch's DistributedDataParallel (DDP)** framework.

---

## Course Overview

Modern deep learning models are large and computationally intensive. Training them on a single GPU can take days or even weeks. This course addresses these challenges by exploring data parallelism techniques for distributing model training across multiple GPUs. The primary focus is on PyTorch’s `DistributedDataParallel`, a widely used API that allows scaling model training efficiently across devices.

By the end of this course, you will be able to:

- Understand the theoretical motivation for multi-GPU training
- Implement and debug training code using PyTorch's DDP
- Work with realistic neural network models and datasets
- Optimize batch size and manage the trade-offs between speed and accuracy

---

## Course Structure

### Lab 1: Introduction and Motivation

**Notebook:** `01_Notebook_Scaling_Training.ipynb`
**Slides:** `lab1p1.pptx`, `lab1p2.pptx`

This lab introduces the motivation for using multiple GPUs in training deep learning models. It covers:

- The growing computational demands of deep learning models
- Differences between Gradient Descent and Stochastic Gradient Descent (SGD)
- The effects of batch size on training dynamics and speed
- Introduction to scaling and performance trends with GPU hardware
- The significance of short iteration times in deep learning research and deployment

---

### Lab 2: Multi-GPU Training with DistributedDataParallel (DDP)

**Notebooks:**
- `01_Notebook_DDP.ipynb`
- `02_Notebook_A_More_Realistic_Model.ipynb`
**Slides:** `lab2p1.pptx`

This lab provides hands-on experience with PyTorch's DistributedDataParallel API:

- Initializing distributed processes and setting up the training environment
- Assigning GPU devices to specific processes
- Wrapping the model with `DistributedDataParallel`
- Efficient dataset partitioning using `DistributedSampler`
- Synchronizing weights and ensuring correct logging and validation in multi-process settings
- Transitioning from toy models to more realistic network architectures

---

### Lab 3: Algorithmic Considerations for Training at Scale

**Notebook:** `02_Notebook_Exercising_Optimization_Strategies.ipynb`
**Slides:** `lab3p1.pptx`

This lab explores advanced considerations when scaling training to multiple GPUs or machines:

- The relationship between batch size and convergence rate
- The accuracy vs. performance trade-offs of large mini-batch training
- Challenges such as the generalization gap and sharp minima
- Mitigation strategies like learning rate scaling, warm-up, and training longer
- Visualizing and interpreting loss landscapes in large-scale models

---

### Final Assessment

**Notebook:** `03_Assessment.ipynb`

A comprehensive assignment where learners apply the techniques learned in previous labs to train a model on multiple GPUs. This includes:

- Writing a complete multi-GPU training loop from scratch
- Choosing appropriate batch size and optimization strategies
- Evaluating model performance under different configurations
- Analyzing training logs for convergence and efficiency

---

## Repository Contents

├── lab1p1.pptx # Intro and scaling motivation
├── lab1p2.pptx # Deeper look at neural nets and loss functions
├── lab2p1.pptx # DDP implementation principles
├── lab3p1.pptx # Batch size scaling and training limits
├── 01_Notebook_Scaling_Training.ipynb # Batch size effects on training
├── 01_Notebook_DDP.ipynb # DDP setup and toy model training
├── 02_Notebook_A_More_Realistic_Model.ipynb # Realistic model training with DDP
├── 02_Notebook_Exercising_Optimization_Strategies.ipynb # Training at scale
├── 03_Assessment.ipynb # Final hands-on exercise


---

## Setup Instructions

To run the notebooks and scripts, you will need:

- Python 3.7+
- PyTorch with CUDA support
- Multiple NVIDIA GPUs
- NCCL backend for communication

### Install Dependencies

```bash
pip install torch torchvision matplotlib
```

### References
- PyTorch Distributed Overview

- Hestness et al., Deep Learning Scaling is Predictable, arXiv:1712.00409

- Keskar et al., On Large-Batch Training for Deep Learning, arXiv:1609.04836

- Kurth et al., Exascale Deep Learning for Climate Analytics, arXiv:1810.01993

- Li et al., Visualizing the Loss Landscape of Neural Nets, arXiv:1712.09913