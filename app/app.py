
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# 1. Configuration

st.set_page_config(
    page_title="CardProfiler",
    page_icon="💳",
    layout="wide"
)

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = "models/cardprofiler_best_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# Adapter ces noms si les labels de ton projet sont différents.
# Si target contient déjà des noms descriptifs, ils seront affichés directement.
SEGMENT_NAMES = {
    0: "Clients Premium",
    1: "Clients à Risque",
    2: "Clients à Faible Activité"
}


def get_segment_name(label):
    # Fonctionne avec des labels numériques ou déjà descriptifs.
    try:
        numeric_label = int(label)
        if numeric_label in SEGMENT_NAMES:
            return SEGMENT_NAMES[numeric_label]
    except (ValueError, TypeError, OverflowError):
        pass

    return str(label)


# 2. Charger le pipeline

st.title(" CardProfiler")
st.subheader("Prédiction du profil comportemental des clients")


try:
    model = load_model()
except Exception as exc:
    st.error(f"Impossible de charger le modèle : {exc}")
    st.stop()


# Récupérer les variables utilisées à l'entraînement.
if hasattr(model, "feature_names_in_"):
    FEATURES = list(model.feature_names_in_)

elif (
    hasattr(model, "named_steps")
    and "scaler" in model.named_steps
    and hasattr(model.named_steps["scaler"], "feature_names_in_")
):
    FEATURES = list(
        model.named_steps["scaler"].feature_names_in_
    )

else:
    # À remplacer par la liste EXACTE de X_train.columns
    # si le modèle n'a pas conservé les noms des variables.
    FEATURES = [
        "BALANCE",
        "CASH_ADVANCE",
        "CREDIT_LIMIT",
        "INSTALLMENTS_PURCHASES",
        "ONEOFF_PURCHASES",
        "PAYMENTS",
        "PURCHASES"
    ]


# 3. Fonction commune de prédiction

def predict_profiles(input_df):

    # Vérifier les colonnes obligatoires
    missing_columns = [
        col for col in FEATURES
        if col not in input_df.columns
    ]

    if missing_columns:
        st.error(
            "Il manque des variables nécessaires au modèle : "
            + ", ".join(missing_columns)
        )
        return

    # Garder uniquement les variables attendues,
    # dans le même ordre que pendant l'entraînement.
    X = input_df.loc[:, FEATURES].copy()

    # Vérifier que les données sont numériques
    for col in FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if X.isna().any().any():
        st.error(
            "Certaines valeurs sont absentes ou non numériques. "
            "Corrige-les avant de lancer la prédiction."
        )
        return

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        st.error("Les données contiennent des valeurs infinies.")
        return

    try:
        predictions = model.predict(X)
    except Exception as exc:
        st.error(f"Erreur pendant la prédiction : {exc}")
        return

    # Afficher les résultats
    results = pd.DataFrame({
        "Segment prédit": [
            get_segment_name(label) for label in predictions
        ]
    })

    st.markdown("### Résultat de la prédiction")

    if len(results) == 1:
        st.success(
            f"Segment du client : **{results.iloc[0, 0]}**"
        )
    else:
        st.dataframe(results, use_container_width=True)

    # Probabilités, si le modèle les propose
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)
        classes = model.classes_

        probability_df = pd.DataFrame(
            probabilities,
            columns=[
                get_segment_name(label) for label in classes
            ]
        )

        if len(results) == 1:
            st.markdown("### Probabilités par segment")

            display_df = pd.DataFrame({
                "Segment": [
                    get_segment_name(label)
                    for label in classes
                ],
                "Probabilité (%)": (
                    probabilities[0] * 100
                ).round(2)
            }).sort_values(
                "Probabilité (%)",
                ascending=False
            )

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True
            )

            st.bar_chart(
                display_df.set_index("Segment")
                ["Probabilité (%)"]
            )

        else:
            # Ajouter les probabilités à chaque prédiction CSV
            results = pd.concat(
                [results, probability_df.add_prefix("Probabilité - ")],
                axis=1
            )

            st.markdown("### Prédictions et probabilités")
            st.dataframe(results, use_container_width=True)

    else:
        st.info(
            "Ce modèle ne fournit pas predict_proba(). "
            "Le segment prédit reste disponible, mais pas les "
            "probabilités par classe."
        )


# 4. Interface : saisie manuelle ou fichier CSV

tab_manual, tab_csv = st.tabs([
    " Saisir un client",
    " Charger un CSV"
])


with tab_manual:

    st.write(
        "Renseigne les variables comportementales du client. "
        "Les valeurs doivent correspondre aux unités et au "
        "format utilisés pendant l'entraînement."
    )

    with st.form("client_form"):

        values = {}
        columns = st.columns(2)

        for i, feature in enumerate(FEATURES):
            with columns[i % 2]:
                values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.4f",
                    key=f"manual_{feature}"
                )

        submitted = st.form_submit_button(
            "Prédire le segment",
            type="primary"
        )

    if submitted:
        client_df = pd.DataFrame([values], columns=FEATURES)
        predict_profiles(client_df)


with tab_csv:

    st.write(
        "Charge un CSV contenant une ou plusieurs lignes clients. "
        "Le fichier doit contenir toutes les variables utilisées "
        "par le modèle."
    )

    uploaded_file = st.file_uploader(
        "Choisir un fichier CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.markdown("### Aperçu des données")
            st.dataframe(
                uploaded_df.head(),
                use_container_width=True
            )

            if st.button(
                "Prédire les segments du fichier",
                type="primary"
            ):
                predict_profiles(uploaded_df)

        except Exception as exc:
            st.error(f"Impossible de lire le CSV : {exc}")
