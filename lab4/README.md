# Lab 4 - Adversarial Training

This laboratory is an introduction to Out Of Distribution Detection and Adversarial Training.
It is divided in three parts:
1. Out Of Distribution Detection: Trained a model to detect if an image is coming from CIFAR-10 (In-Distribution) or a subset of CIFAR-100 including only 5 classes representing humans (Out-Of-Distribution).
2. Adversarial Attacks: Showed how to generate adversarial examples using the Fast Gradient Sign Method (FGSM), and trained a robust model using adversarial training.  
3. JARN: Implemented JARN (Jacobian Adversarially Regularized Networks for Robustness), a method to train robust models using adversarial training.

[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=plastic&logo=jupyter&logoColor=white)](./Lab4_OOD.ipynb)

An implementation of all three parts can be found in the [Jupyter Notebook](./Lab4_OOD.ipynb).

## 1. Out Of Distribution Detection

![Out Of Distribution Detection histogram](./images/ood_detection_histogram.png)

## 2. Adversarial Attacks

![FGSM example](./images/fgsm_example.png)

![FGSM epsilons](./images/fgsm_epsilons.png)

## 3. JARN

![JARN example](./images/jarn_example.png)

