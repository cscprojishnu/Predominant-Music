# Predominant-Music

# 🎼 Predominant Instrument Recognition in Polyphonic Music using CNN + INN Ensemble

This repository contains the implementation of our research titled:

> **Dynamic Feature Learning with Involution and Convolution for Predominant Instrument Recognition in Polyphonic Music**  
> Published in: *Circuits, Systems, and Signal Processing* (Springer, 2025)  
> DOI: [10.1007/s00034-025-03111-y](https://doi.org/10.1007/s00034-025-03111-y)

📍 Authors: [C. R. Lekshmi](mailto:cr_lekshmi@cb.amrita.edu), [Jishnu Teja Dandamudi](mailto:djishnuteja2006@gmail.com)  
🏫 Amrita School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore, India

---

## 📌 Abstract

This work proposes a **hybrid deep learning framework** for predominant instrument recognition in polyphonic music using:
- **Convolutional Neural Networks (CNNs)** for capturing global frequency structures
- **Involution Neural Networks (INNs)** for learning localized spatial patterns

A **soft-voting ensemble** strategy combines their predictions, achieving state-of-the-art results on the **IRMAS dataset**, with fewer parameters and improved efficiency.

---

## 🧠 Key Contributions

- ✅ A novel ensemble model combining CNN and INN
- ✅ Elimination of sliding window aggregation techniques
- ✅ 24.59% improvement in micro-F1 and 27.77% in macro-F1 over Han et al.
- ✅ Lightweight architecture (7k–641k params vs. 1.4M in Han’s model)

---

## 🗃 Dataset

- **IRMAS Dataset** (Instrument Recognition in Musical Audio Signals)  
  [Access Here](https://www.upf.edu/web/mtg/irmas)  
  - 6705 training samples (3 seconds each)
  - 2874 test samples (5–20 seconds)

---

## 🏗️ Architecture

### 🌀 CNN Module
- Extracts global harmonic patterns
- ~641k parameters

### 🌀 INN Module
- Learns position-specific spatial kernels
- ~7k parameters

### 🔀 Ensemble CI (Convolution + Involution)
- Soft-voting on predicted class probabilities
- Combines the strengths of both CNN and INN

---

## 🧪 Experimental Setup

- **Feature Extraction**: 224 Mel-filterbanks, FFT=8192, hop=441
- **Training**: 200 epochs, Adam optimizer, categorical cross-entropy
- **Validation Split**: 80-20
- **Platform**: Google Colab
- **Evaluation Metrics**: Micro-F1, Macro-F1, Precision, Recall

---

## 📊 Performance Summary

| Model         | Params | Micro F1 | Macro F1 |
|---------------|--------|----------|----------|
| Han et al.    | 1.4M   | 0.60     | 0.50     |
| CNN (Ours)    | 641k   | 0.74     | 0.64     |
| INN (Ours)    | 7k     | 0.75     | 0.65     |
| **Ensemble CI** | -    | **0.76** | **0.69** |

---
