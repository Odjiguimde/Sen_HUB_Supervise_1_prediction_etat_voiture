"""
Application Streamlit — Prédiction de l'état d'un véhicule (Occasion / Venant)
Déploiement du meilleur modèle issu du notebook "Clas_7_Modeles_Car_Data".

Fichiers requis dans le même dossier que ce script :
    - best_model.joblib   (meilleur des 7 modèles, sélectionné dans le notebook)
    - encoders.joblib     (LabelEncoder : [Marque, Transmission, Quartier])
    - uniques.joblib      (valeurs uniques : [Marque, Transmission, Quartier, Etat])
    - scaler.joblib       (MinMaxScaler entrainé sur x)

Lancement en local :  streamlit run app_car_data.py
"""

import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st


st.set_page_config(
    page_title="Prédiction de l'état d'un véhicule",
    page_icon="🚗",
    layout="centered",
)

DESCRIPTION = (
    "Ce modèle de machine learning prédit si un véhicule est **Occasion** ou "
    "**Venant** à partir de la marque, l'année, la transmission, le quartier "
    "et le prix."
)

# Ordre des variables prédictrices tel qu'utilisé lors de l'entrainement (x)
FEATURES_ORDER = ["Marque", "Année", "Transmission", "Quartier", "Prix"]


@st.cache_resource
def load_artifacts():
    encoders = jb.load("encoders.joblib")     # [enc_marque, enc_transmission, enc_quartier]
    uniques = jb.load("uniques.joblib")       # [Marque, Transmission, Quartier, Etat]
    scaler = jb.load("scaler.joblib")         # MinMaxScaler
    best_model = jb.load("best_model.joblib") # meilleur modèle des 7
    return encoders, uniques, scaler, best_model


encoders, uniques, scaler, best_model = load_artifacts()
class_names = uniques[-1]  # ['Occasion', 'Venant']


def Pred_func(marque, annee, transmission, quartier, prix):
    marque_enc = encoders[0].transform([marque])[0]
    transmission_enc = encoders[1].transform([transmission])[0]
    quartier_enc = encoders[2].transform([quartier])[0]

    x_new = np.array([marque_enc, annee, transmission_enc, quartier_enc, prix])
    x_new = x_new.reshape(1, -1)
    x_new = scaler.transform(x_new)
    y_pred = best_model.predict(x_new)
    return class_names[int(y_pred[0])]


def Pred_func_csv(file):
    df = pd.read_csv(file)
    predictions = []
    for row in df[FEATURES_ORDER].values:
        y_pred = Pred_func(*row)
        predictions.append(y_pred)
    df["Etat"] = predictions
    return df


st.title("🚗 Prédiction de l'état d'un véhicule")
st.caption(f"Meilleur modèle chargé : **{type(best_model).__name__}**")

onglet1, onglet2 = st.tabs(["Prédiction simple", "Prédiction multiple"])

# ----------------------------- Onglet 1 -------------------------------
with onglet1:
    st.subheader("Prédire l'état d'un véhicule avec une entrée")
    st.write(DESCRIPTION)

    with st.form("formulaire_simple"):
        col1, col2 = st.columns(2)
        with col1:
            marque = st.selectbox("Marque", options=list(uniques[0]))
            annee = st.number_input("Année", min_value=1980, max_value=2026, value=2015, step=1)
            transmission = st.selectbox("Transmission", options=list(uniques[1]))
        with col2:
            quartier = st.selectbox("Quartier", options=list(uniques[2]))
            prix = st.number_input("Prix", value=0.0, step=100000.0, format="%.2f")

        soumettre = st.form_submit_button("Prédire", type="primary")

    if soumettre:
        try:
            resultat = Pred_func(marque, annee, transmission, quartier, prix)
            st.success(f"**État du véhicule :** {resultat}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# ----------------------------- Onglet 2 -------------------------------
with onglet2:
    st.subheader("Prédire l'état de plusieurs véhicules")
    st.write(DESCRIPTION)
    st.caption(
        "Le fichier CSV doit contenir, dans cet ordre, les colonnes : "
        "Marque, Année, Transmission, Quartier, Prix."
    )

    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if fichier is not None:
        try:
            with st.spinner("Prédictions en cours…"):
                df_resultat = Pred_func_csv(fichier)

            st.success(f"{len(df_resultat)} prédiction(s) effectuée(s).")
            st.dataframe(df_resultat, use_container_width=True)

            st.download_button(
                label="⬇️ Télécharger le fichier CSV",
                data=df_resultat.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")
