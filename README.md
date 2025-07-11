# The Yeole Compression Criterion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15864512.svg)](https://doi.org/10.5281/zenodo.15864512)

**A Storage-Efficient Bound for PCA-Based Dimensionality Reduction**  
By [Om Yeole](https://orcid.org/0009-0001-9061-9725)  
Indian Institute of Technology Madras  
Released under [CC-BY 4.0 License](https://creativecommons.org/licenses/by/4.0/)

---

## 🔍 Overview

Principal Component Analysis (PCA) is often used to reduce dimensionality by maximizing retained variance or minimizing reconstruction error. But what if PCA actually **increases memory usage** instead of saving it?

This repo introduces the **Yeole Compression Criterion** — a simple, provable bound to determine whether PCA truly compresses your data in terms of **element-wise memory storage**.

---

## 📄 Paper & Citation

**DOI:** https://doi.org/10.5281/zenodo.15864512  
**Citation:**

> Yeole, Om. *The Yeole Compression Criterion: A Practical Bound for PCA-Based Dimensionality Reduction*. Zenodo. July 11, 2025. https://doi.org/10.5281/zenodo.15864512

---

## 📈 Highlights

- **Yeole Ratio**:  
  K = (N × D) / (N + D)
  A tight upper bound for choosing the number of PCA components (M) that still save memory.

- **Criterion**:  
  - If K < 1: PCA does not compress — use full original data.  
  - If K > 1: Retain at most floor(K) components for memory-efficient compression.

- **Fully validated** using experiments on the MNIST dataset  
- **Plots and reconstruction visuals** included

---

## 📦 Repo Contents

- `yeole_compression.py` — Full python script
  
---

## 🧪 Run the Experiment

Requires Python ≥ 3.7  
Install required packages:

```bash
pip install numpy pandas scikit-learn matplotlib
