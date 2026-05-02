# 🚀 Dashboard Streamlit — Détection de Fraude Bancaire
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# ============================================
# Configuration de la page
# ============================================
st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="🔍",
    layout="wide"
)

# ============================================
# Titre principal
# ============================================
st.title("🔍 Détection de Fraude Bancaire")
st.markdown("**Projet Data Science — Master 1 IA/Data Science**")
st.markdown("---")

# ============================================
# Chargement des données
# ============================================
@st.cache_data
def charger_donnees():
    df = pd.read_csv('data/creditcard.csv')
    df['Hour']       = (df['Time'] / 3600) % 24
    df['Amount_Log'] = np.log1p(df['Amount'])
    return df

# ============================================
# Entraînement du modèle
# ============================================
@st.cache_resource
def entrainer_modele(df):
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()
    X_train_scaled['Amount'] = scaler.fit_transform(
        X_train[['Amount']]
    )
    X_test_scaled['Amount'] = scaler.transform(
        X_test[['Amount']]
    )

    smt = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smt.fit_resample(
        X_train_scaled, y_train
    )

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    model.fit(X_train_res, y_train_res)

    return model, scaler, X_test_scaled, y_test

# ============================================
# Sidebar — Navigation
# ============================================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Aller à :", [
    "📊 Exploration des données",
    "🤖 Performance du modèle",
    "🔍 Analyser une transaction"
])

# ============================================
# Chargement
# ============================================
with st.spinner("⏳ Chargement des données..."):
    df = charger_donnees()

# ============================================
# PAGE 1 — Exploration
# ============================================
if page == "📊 Exploration des données":
    st.header("📊 Exploration des Données")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total transactions", f"{len(df):,}")
    col2.metric("Fraudes détectées", f"{df['Class'].sum():,}")
    col3.metric("Taux de fraude", f"{df['Class'].mean()*100:.3f}%")
    col4.metric("Montant moyen", f"{df['Amount'].mean():.2f}€")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔴 Déséquilibre des classes")
        fig, ax = plt.subplots(figsize=(6, 4))
        df['Class'].value_counts().plot(
            kind='bar', ax=ax,
            color=['#2ecc71', '#e74c3c'],
            edgecolor='black'
        )
        ax.set_xticklabels(
            ['Normale', 'Fraude'], rotation=0
        )
        ax.set_ylabel('Nombre')
        st.pyplot(fig)

    with col2:
        st.subheader("⏱️ Fraudes par heure")
        fig, ax = plt.subplots(figsize=(6, 4))
        fraudes = df[df['Class'] == 1]
        ax.hist(fraudes['Hour'], bins=24,
                color='#e74c3c', edgecolor='white')
        ax.set_xlabel('Heure')
        ax.set_ylabel('Nombre de fraudes')
        st.pyplot(fig)

# ============================================
# PAGE 2 — Performance du modèle
# ============================================
elif page == "🤖 Performance du modèle":
    st.header("🤖 Performance du Modèle XGBoost")

    with st.spinner("⏳ Entraînement en cours (5-10 min)..."):
        model, scaler, X_test_scaled, y_test = entrainer_modele(df)

    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    col1, col2, col3 = st.columns(3)
    col1.metric("AUC-ROC", f"{auc:.4f}")
    col2.metric("Fraudes détectées",
                f"{(y_pred[y_test==1]==1).sum()}/98")
    col3.metric("Faux Positifs",
                f"{(y_pred[y_test==0]==1).sum()}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Matrice de Confusion")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d',
                    cmap='Oranges', ax=ax,
                    xticklabels=['Normale', 'Fraude'],
                    yticklabels=['Normale', 'Fraude'])
        ax.set_ylabel('Réel')
        ax.set_xlabel('Prédit')
        st.pyplot(fig)

    with col2:
        st.subheader("📈 Courbe ROC")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color='#e74c3c', lw=2,
                label=f'XGBoost (AUC = {auc:.4f})')
        ax.plot([0,1],[0,1],'k--', label='Aléatoire')
        ax.set_xlabel('Faux Positifs')
        ax.set_ylabel('Vrais Positifs')
        ax.legend()
        st.pyplot(fig)

# ============================================
# PAGE 3 — Analyser une transaction
# ============================================
elif page == "🔍 Analyser une transaction":
    st.header("🔍 Analyser une Transaction")

    with st.spinner("⏳ Chargement du modèle..."):
        model, scaler, X_test_scaled, y_test = entrainer_modele(df)

    st.info("👇 Ajuste les valeurs et clique sur Analyser")

    col1, col2, col3 = st.columns(3)
    with col1:
        montant = st.slider("💰 Montant (€)", 0.0, 5000.0, 100.0)
    with col2:
        heure = st.slider("⏰ Heure", 0, 23, 12)
    with col3:
        v14 = st.slider("V14", -10.0, 5.0, 0.0)

    if st.button("🔍 Analyser cette transaction"):
        sample = X_test_scaled.iloc[0:1].copy()
        sample['Amount']     = scaler.transform([[montant]])[0][0]
        sample['Hour']       = heure
        sample['Amount_Log'] = np.log1p(montant)
        sample['V14']        = v14

        proba = model.predict_proba(sample)[0][1]
        pred  = model.predict(sample)[0]

        st.markdown("---")
        if pred == 1:
            st.error(f"🚨 FRAUDE DÉTECTÉE — Score : {proba:.2%}")
        else:
            st.success(f"✅ TRANSACTION NORMALE — Score : {proba:.2%}")

        st.progress(float(proba))