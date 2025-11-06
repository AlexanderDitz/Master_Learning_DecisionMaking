# Computational Modeling of Individual Learning Dynamics in Affective Disorders

This repository contains the code, data structure, and analyses for the thesis **“Modeling Individual Learning Dynamics in Healthy and Affective Populations Using Computational and Deep Learning Approaches.”**  
The project investigates how individuals with **depression** or **bipolar disorder** differ from **healthy controls** in their learning and decision-making behavior.

---

## 🧠 Overview

We combine **computational modeling** with **unsupervised machine learning** to explore latent patterns in learning dynamics:

- **Generalized Q-Learning (GQL)** to capture reinforcement learning parameters  
- **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTM)** models to learn temporal dependencies in behavioral data  
- **K-Means clustering** and **t-SNE** visualization to identify distinct learning profiles across individuals  

These models were trained and evaluated on behavioral datasets collected from healthy, depressed, and bipolar participants performing probabilistic learning tasks.

---

## 📂 Repository Structure

├── data/ # Preprocessed behavioral data (or data loading scripts)
├── models/
│ ├── gql_model.py # Implementation of the GQL model
│ ├── rnn_model.py # Recurrent neural network architecture
│ ├── lstm_model.py # LSTM architecture
│ └── utils/ # Helper functions and shared modules
├── analysis/
│ ├── clustering.ipynb # K-Means clustering and t-SNE visualization
│ ├── model_comparison.ipynb# Model fitting and evaluation
│ └── parameter_analysis.ipynb
├── figures/ # Generated figures and plots
├── results/ # Model outputs, clustering results
├── requirements.txt # Python dependencies
└── README.md # This file


---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

### 2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

### 3. Install dependencies
pip install -r requirements.txt

