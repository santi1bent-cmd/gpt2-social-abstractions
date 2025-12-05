# gpt2-social-abstractions
Analysis and code for probing how GPT-2 Medium represents intentionality, fairness, and deception using linear probes and activation-space geometry.
# Probing Social Abstractions in GPT-2: Intentionality, Fairness, and Deception

This repository contains the full code and figures accompanying the paper:

**Santi Bent (2025). _Probing Social Abstractions in GPT-2: Intentionality, Fairness, and Deception._**

The project investigates whether GPT-2 Medium represents three socially meaningful abstractions—**intentionality**, **fairness**, and **deception**—in ways that are linearly accessible in its hidden activation space. Using a probing pipeline built on linear classifiers, activation analysis, and representation geometry, the study evaluates how these concepts emerge (or fail to emerge) across the model’s 24 layers.

---

##  Overview

This repo provides:

- **Activation extraction** for every layer of GPT-2 Medium  
- **Linear probing** using logistic regression  
- **Separation score computation** (centroid distance vs. within-class variance)  
- **PCA visualization tools** for hidden-state geometry  
- **Scripts for reproducing all figures** used in the paper  

The goal is to support full reproducibility and allow further exploration of how LLM's encode socially grounded abstractions.

Datasets Used
  Intentionality: ToMI-NLI
  Fairness: CrowS-Pairs
  Deception: TruthfulQA (multiple-choice version)

If you use this code or build on this work, please cite:
  @misc{bent2025socialabstractions,
    title   = {Probing Social Abstractions in GPT-2: Intentionality, Fairness, and Deception},
    author  = {Bent, Santi},
    year    = {2025},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CL}
  }

---



