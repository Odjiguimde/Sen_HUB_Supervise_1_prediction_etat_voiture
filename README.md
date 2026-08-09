# Prédiction de l'état d'un véhicule (Occasion / Venant)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://appcardatapy-cuqwdhfehxnfhlbnvtwps7.streamlit.app/)

Projet de classification supervisée appliqué à des données de véhicules en vente à Dakar, scrapées depuis [expat-dakar.com](https://www.expat-dakar.com/voitures/dakar).

## 🌐 Démo en ligne

👉 **[Essayer l'application ici](https://appcardatapy-cuqwdhfehxnfhlbnvtwps7.streamlit.app/)**

## Objectif

Prédire si un véhicule est **Occasion** ou **Venant** à partir de :
- Marque
- Année
- Transmission
- Quartier
- Prix

## Données

| Variable | Type | Description |
|---|---|---|
| Marque | Catégorielle | Marque du véhicule |
| Année | Numérique | Année de mise en circulation |
| Transmission | Catégorielle | Boîte manuelle / automatique |
| Quartier | Catégorielle | Zone de vente à Dakar |
| Prix | Numérique | Prix de vente (FCFA) |
| Etat (cible) | Catégorielle | Occasion / Venant |

## Modélisation

7 algorithmes de classification sont entrainés et comparés, chacun optimisé via `GridSearchCV` :

1. K-Nearest Neighbors (KNN)
2. Logistic Regression
3. Support Vector Machine (SVM)
4. Decision Tree
5. Random Forest
6. Gradient Boosting
7. XGBoost

**Pipeline** : nettoyage des adresses → encodage (`LabelEncoder`) → traitement des outliers (IQR + `KNNImputer`) → normalisation (`MinMaxScaler`) → split train/val/test (80/10/10) → recherche d'hyperparamètres → évaluation (Accuracy, F1, Precision, Recall) → sélection automatique du meilleur modèle → déploiement.

## Résultats

Le meilleur modèle est sélectionné automatiquement en fin de notebook selon le F1-score sur les données de validation, puis sauvegardé (`best_model.joblib`).

## Déploiement

Application Streamlit (`app_car_data.py`) permettant :
- une prédiction manuelle via formulaire,
- une prédiction en lot via import CSV (colonnes : `Marque, Année, Transmission, Quartier, Prix`).

### Lancer en local

```bash
pip install -r requirements.txt
streamlit run app_car_data.py
```

## Structure du projet
├── Clas_7_Modeles_Car_Data.ipynb # Notebook d'entrainement (7 modèles)

├── app_car_data.py # Application de déploiement Streamlit

├── best_model.joblib # Meilleur modèle entrainé

├── encoders.joblib # Encodeurs des variables catégorielles

├── uniques.joblib # Valeurs uniques (pour les listes déroulantes)

├── scaler.joblib # Normaliseur (MinMaxScaler)

├── requirements.txt

└── README.md

## Stack technique

Python · pandas · scikit-learn · XGBoost · Plotly · Streamlit

## Auteur

Oumaro Titans DJIGUIMDE — ESMT Dakar, filière Data Engineering & Intelligence Artificielle
