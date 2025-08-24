# HierachicalNets: Multi-level Hierarchical Classification of Yoga Poses

This repository contains the official implementation of the paper:

**HierachicalNets: Multi-level Hierarchical Classification of Yoga Poses**

---

## 📌 Overview
We propose **HierachicalNets**, a set of hierarchical vision transformer architectures (based on CoAtNet and MaxViT) for fine-grained yoga pose classification.  

Key innovations include:
- **Hierarchical supervision** at multiple semantic levels  
  (Level 1: 6 classes, Level 2: 20 classes, Level 3: 82 classes)  
- **Hybrid backbones** (convolution + self-attention)  
- **Stage-wise auxiliary losses** to improve gradient flow and generalization  

Our approach **significantly outperforms prior state-of-the-art** models on the **Yoga-82 dataset**, while also being faster at inference.

---

## 📊 Dataset
We use the **Yoga-82 dataset** (Verma et al., CVPRW 2020).  

- Publicly available here: [Yoga-82 Dataset](https://sites.google.com/view/yoga-82/home)  
- The dataset has ~28,478 images across 82 yoga poses, structured hierarchically.  

