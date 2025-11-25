# EuroSAT Classification Project

## Project Overview  
This repository contains the code and resources for the **EuroSAT Classification** project for ECEN 785.  
The goal of this project is to train, evaluate, and compare deep learning models for land-use classification using the EuroSAT dataset. The code includes data processing, model training, and benchmarking of architectures.

---

## Team  
**Team Name:** Group 17  
**Team Members:**  
- Member 1 – Praroop Chanda  
- Member 2 – Aditya Rao Ghodke  
- Member 3 – Deepmoy Hazra  
- Member 4 – Yuhan Fu

---

### Files & Notebooks  
- `eurosat.ipynb` — main notebook: Jupyter notebook containing the full workflow: data loading, preprocessing, model training, and evaluation. 
- `Train_Pipeline_1.png` — Image that shows illustration of the training pipeline, imported in eurosat.ipynb
- `Resnet.png`, `Resnet_18_custom_head.png` — architecture diagrams, imported in eurosat.ipynb
- `model_checkpoint_1e-05.pkt` — Saved trained model weights used for evaluation and inference.
- `test.py` - Script used to load the trained model and run inference on test images or directories.

---

## How to Run  
1. Clone this repository:  
   ```bash
   git clone https://github.com/YourUsername/EuroSAT_Classification_Project_ECEN_785.git
   cd EuroSAT_Classification_Project_ECEN_785
   ```
2. Install Dependencies
   ``` bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook (Optional: To train the whole model and save the model output - run this block of code)
   ``` bash
   jupyter notebook eurosat.ipynb
   ```
4. Run the Test Script
   ``` bash
   python test.py
   ```

## 🔬 Research Extension: Generative AI for Data Augmentation

As part of a research extension, we explored **Generative AI techniques** to increase the training dataset for the EuroSAT classification task.  
Using a **Variational Autoencoder (VAE)**, we generated additional synthetic images. T

This extension shows how generative models can be applied to **enhance datasets** in remote sensing and land-use classification projects.  

For details, see the code and experiments in the repository link:
Link: https://github.com/yuhanfu11/ecen758_research_extension
  
