# HierachicalNets: Multi-level Hierarchical Classification of Yoga Poses

This repository contains the official implementation of the paper:

**HierachicalNets: Multi-level Hierarchical Classification of Yoga Poses**

---

## 📌 Overview
We propose **HierachicalNets**, a set of hierarchical vision transformer architectures (based on CoAtNet and MaxViT) for fine-grained yoga pose classification.  
The models integrate:
- **Hierarchical supervision** at multiple semantic levels (Level 1: 6 classes, Level 2: 20 classes, Level 3: 82 classes)  
- **Hybrid backbones** (convolution + self-attention)  
- **Stage-wise auxiliary losses** to improve gradient flow and generalization  

Our approach surpasses state-of-the-art performance on the **Yoga-82 dataset**.

---

## 📊 Dataset
- We use the **Yoga-82 dataset** (Verma et al., CVPRW 2020).  
- Publicly available here: [Yoga-82 Dataset](https://sites.google.com/view/yoga-82/home).  
- Ensure the dataset is downloaded and organized as follows:

