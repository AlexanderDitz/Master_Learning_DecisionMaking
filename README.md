# Computational Modeling of Individual Learning Dynamics in Affective Disorders

This repository contains the code, data structure, and analyses for the thesis  
**“Discovering individual differences in learning dynamics between healthy and clinical populations.”**  

The current project investigates how individuals with **depression** or **bipolar disorder** differ from **healthy controls** in their learning and adaptive decision-making behavior.

---

## 🧠 Overview

We combine **computational modeling** with **unsupervised machine learning** to explore latent patterns in learning dynamics:

- **Generalized Q-Learning (GQL):** captures reinforcement learning parameters  
- **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTM):** learn temporal dependencies in behavioral data  
- **SPICE:** extracts behavioral equations (coefficients)
- **K-Means clustering** and **t-SNE visualization:** identify distinct learning profiles across individuals  

Models are trained and fitted on behavioral data (by Dezfouli et al. (2019)) collected from healthy, depressed, and bipolar participants performing probabilistic learning tasks.

---

## 📂 Repository Structure

```
├── data/                # Behavioral and synthetic datasets
├── params/              # Model definitions and training scripts
│   ├── gql_model.py
│   ├── rnn_model.py
│   ├── lstm_model.py
│   └── utils/
├── analysis/            # Extraction, clustering, and plotting scripts
│   ├── extract_rnn_hidden_state.py
│   ├── extract_lstm_hidden_state.py
│   ├── extract_gql_parameters.py
│   ├── cluster_analysis_rnn.py
│   ├── cluster_analysis_lstm.py
│   ├── cluster_analysis_gql.py
│   └── plot_2d_vector_flow_rnn.py
├── figures/             # Generated figures and plots
├── results/             # Model outputs, clustering results
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create a virtual environment and activate it

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏗️ Analysis Pipeline

1. **Data preprocessing:** Scripts in `data/`
2. **Model training:** Train GQL, RNN, and LSTM models on real data (`models/`)
3. **Parameter/hidden state extraction:**  
   - Per-participant (real data) for clustering  
   - Per-trial (synthetic data) for vector field analysis
4. **Clustering & visualization:** K-Means, t-SNE (`analysis/`)
5. **Statistical analysis:** Compare clusters, visualize group differences

---

## 💻 Example Usage

**Extract RNN hidden states per trial:**
```bash
python analysis/extract_rnn_hidden_state.py
```

**Extract LSTM hidden states per trial:**
```bash
python analysis/extract_lstm_hidden_state.py
```

**Extract GQL parameters and Q-values:**
```bash
python analysis/extract_gql_parameters.py
```

**Run clustering analysis (RNN example):**
```bash
python analysis/cluster_analysis_rnn.py
```

**Plot vector field for RNN:**
```bash
python plot_2d_vector_flow_rnn.py
```

---

## 📢 Contact

Alexander Ditz, 2025  
[aditz@uni-osnabrueck.de]

---

## 📄 License

[Specify license if applicable]