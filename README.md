# 🔍 Détection de Fraude Bancaire par Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange)
![SHAP](https://img.shields.io/badge/SHAP-Interpretability-red)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 🎯 Contexte Métier

La fraude bancaire représente des milliards de pertes annuelles 
pour les institutions financières. Ce projet construit un système 
de détection automatique des transactions frauduleuses en utilisant 
le Machine Learning, capable d'analyser des milliers de transactions 
en temps réel.

## 📊 Résultats Clés

| Modèle | AUC-ROC | Precision | Recall | F1-Score |
|--------|---------|-----------|--------|----------|
| Régression Logistique | 0.9706 | 0.06 | 0.91 | 0.10 |
| Random Forest | 0.9721 | 0.86 | 0.83 | 0.84 |
| **XGBoost** ✅ | **0.9782** | **0.42** | **0.87** | **0.56** |

> **Modèle final choisi : XGBoost** — meilleur AUC-ROC et meilleur Recall

## 🔑 Insights Principaux

- 🚨 Seulement **0.17%** des transactions sont frauduleuses
  → gestion du déséquilibre avec **SMOTETomek**
- 🌙 Pic de fraudes à **2h du matin** — les fraudeurs agissent
  la nuit
- 💰 Les fraudes ont souvent des **montants très faibles** (médiane
  9.25€) — technique de "card testing"
- 🔍 **V14 et V4** sont les variables les plus importantes
  selon SHAP

## 🛠️ Technologies Utilisées
Python 3.10+
├── pandas / numpy          → Manipulation des données
├── matplotlib / seaborn    → Visualisation
├── scikit-learn            → Pipeline ML
├── xgboost                 → Modèle final
├── imbalanced-learn        → Gestion du déséquilibre
├── shap                    → Interprétabilité
└── streamlit               → Dashboard interactif

## 📁 Structure du Projet
fraud-detection-ml/
├── data/                   → Dataset (non inclus)
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preparation_donnees.ipynb
│   ├── 03_modelisation.ipynb
│   └── 04_interpretabilite.ipynb
├── reports/                → Graphiques générés
├── src/                    → Scripts Python
└── README.md

## 📂 Dataset

🔗 [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Comment télécharger le dataset ?
1. Crée un compte sur [kaggle.com](https://www.kaggle.com)
2. Clique sur le lien ci-dessus → **Download**
3. Place `creditcard.csv` dans le dossier `data/`

## 🚀 Installation

```bash
# Cloner le repo
git clone https://github.com/ange-Kouassi/fraud-detection-ml.git
cd fraud-detection-ml

# Installer les dépendances
pip install -r requirements.txt

# Lancer Jupyter
jupyter notebook
```

## 📈 Méthodologie

1. **Exploration (EDA)** → Analyse du déséquilibre, 
   distribution temporelle, analyse des montants
2. **Préparation** → Feature Engineering, normalisation, 
   SMOTETomek
3. **Modélisation** → Comparaison de 3 modèles ML
4. **Interprétabilité** → Analyse SHAP globale et individuelle

## 👤 Auteur

**Ange Emmanuel Zaibaihi KOUASSI** — Étudiant Master 1 Data Science / IA  
📧 zaibaihi@gmail.com  

## 🌐 Dashboard Live

👉 **[Voir le dashboard en ligne](https://fraud-detection-ml-bt478ljuqk2n6ygpsnyuek.streamlit.app/)**

> ⚠️ Le fichier CSV n'est pas inclus dans ce repo (trop volumineux).

> Il doit être téléchargé séparément depuis Kaggle.

