"""Dashboard Streamlit : prédiction de la réponse à une campagne marketing.

Fichiers attendus dans le dossier models/ :
- modele_final.joblib  : pipeline scikit-learn (transformation log + imputation + scaling + régression logistique)
- colonnes.joblib      : liste ordonnée des 28 colonnes attendues par le modèle
- scores_base.joblib   : scores de tous les clients de la base (pour situer un client dans la base)
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent / "models"

st.set_page_config(page_title="Réponse aux campagnes marketing", layout="wide")


@st.cache_resource
def charger():
    modele = joblib.load(BASE / "modele_final.joblib")
    colonnes = joblib.load(BASE / "colonnes.joblib")
    scores_base = joblib.load(BASE / "scores_base.joblib")
    return modele, colonnes, scores_base


modele, colonnes, scores_base = charger()

page = st.sidebar.radio(
    "Navigation",
    ["Vue d'ensemble", "Ce qui explique la réponse", "Simulateur client"],
)
st.sidebar.caption(
    "Données : campagne marketing 2012-2014 (Kaggle). "
    "Projet de démonstration, non destiné à un usage en production."
)

# ---------------------------------------------------------------- Page 1
if page == "Vue d'ensemble":
    st.title("Prédire qui répondra à une campagne marketing")
    st.write(
        "Objectif : à partir du profil et du comportement d'achat d'un client, "
        "estimer s'il va accepter l'offre de la dernière campagne. "
        "Seuls 14,9 % des clients répondent, ce qui rend l'accuracy trompeuse : "
        "la métrique de référence est le **PR-AUC**."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PR-AUC (test)", "0,64", help="Intervalle de confiance à 95 % : 0,52 à 0,76")
    c2.metric("Référence naïve", "0,15", help="Un modèle sans information obtient la proportion de réponses")
    c3.metric("Recall (test)", "80 %", help="Part des clients répondeurs repérés (seuil 0,5)")
    c4.metric("Precision (test)", "47 %", help="Part des clients signalés qui répondent vraiment (seuil 0,5)")
    st.caption(
        "Mesuré sur un jeu de test de 447 clients dont 66 répondeurs : "
        "l'incertitude est large (IC 95 % du PR-AUC : 0,52 à 0,76)."
    )

    st.subheader("Comparaison des modèles (validation croisée, jeu d'entraînement)")
    comparaison = pd.DataFrame(
        {
            "Modèle": [
                "Référence naïve",
                "Random Forest",
                "SVM (réglages par défaut)",
                "XGBoost",
                "Régression logistique",
                "Régression logistique + transformation log",
            ],
            "PR-AUC": [0.149, 0.630, 0.646, 0.649, 0.653, 0.681],
        }
    ).set_index("Modèle")
    st.bar_chart(comparaison)
    st.write(
        "Les écarts entre les modèles les plus performants sont de l'ordre de la variation "
        "entre plis (≈ 0,03) : aucun n'est démontré meilleur. La **régression logistique** "
        "a été retenue pour sa simplicité et parce que ses coefficients sont lisibles. "
        "Ce qui a le plus amélioré les scores : la transformation logarithmique des "
        "montants très asymétriques, pas le choix d'un modèle plus complexe."
    )

# ---------------------------------------------------------------- Page 2
elif page == "Ce qui explique la réponse":
    st.title("Ce qui explique la réponse")
    st.write(
        "Importance par groupes de variables : baisse du PR-AUC quand on brouille "
        "les valeurs du groupe dans les données de validation. Plus la baisse est grande, "
        "plus le groupe compte pour le modèle."
    )
    importance = pd.DataFrame(
        {
            "Baisse du PR-AUC": {
                "Campagnes passées": 0.323,
                "Récence": 0.109,
                "Foyer (enfants, couple)": 0.093,
                "Ancienneté du client": 0.087,
                "Canaux d'achat": 0.071,
                "Profil (revenu, âge, études)": 0.037,
                "Dépenses par produit": 0.025,
                "Plainte": -0.001,
            }
        }
    )
    st.bar_chart(importance)
    st.markdown(
        "- **Le meilleur prédicteur d'une réponse est d'avoir déjà répondu** : les clients "
        "ayant accepté une campagne passée répondent 3 à 5 fois plus.\n"
        "- Viennent ensuite la **récence** du dernier achat, la composition du **foyer** et "
        "l'**ancienneté** du client. L'ordre entre ces trois groupes n'est pas fiable.\n"
        "- Les **dépenses** ressortent faiblement parce qu'elles sont très liées au revenu "
        "et aux achats par canal : cette méthode sous-estime les groupes redondants entre eux.\n"
        "- Les **plaintes** n'apportent aucune information."
    )
    st.info(
        "Profil du répondeur : client ancien, ayant acheté récemment, ayant déjà accepté "
        "des campagnes, plutôt sans conjoint ni adolescents au foyer, plutôt diplômé."
    )

# ---------------------------------------------------------------- Page 3
else:
    st.title("Simulateur client")
    st.write("Renseignez le profil d'un client pour obtenir son score de propension à répondre.")

    niveaux = {"Basic": 0, "Graduation": 1, "2n Cycle": 2, "Master": 2, "PhD": 3}

    with st.form("simulateur"):
        st.subheader("Historique des campagnes")
        st.caption("Cochez les campagnes que le client a acceptées.")
        cols = st.columns(6)
        cmp1 = cols[0].checkbox("Campagne 1")
        cmp2 = cols[1].checkbox("Campagne 2")
        cmp3 = cols[2].checkbox("Campagne 3")
        cmp4 = cols[3].checkbox("Campagne 4")
        cmp5 = cols[4].checkbox("Campagne 5")
        plainte = cols[5].checkbox("Plainte (2 ans)")

        st.subheader("Profil")
        c1, c2, c3 = st.columns(3)
        revenu = c1.number_input("Revenu annuel du foyer", 0, 200000, 51000, step=1000)
        age = c2.slider("Âge", 18, 90, 45)
        education = c3.selectbox("Niveau d'études", list(niveaux), index=1)
        c1, c2, c3 = st.columns(3)
        couple = c1.checkbox("Vit en couple (marié ou en union)")
        kidhome = c2.number_input("Jeunes enfants au foyer", 0, 5, 0)
        teenhome = c3.number_input("Adolescents au foyer", 0, 5, 0)

        st.subheader("Relation avec l'enseigne")
        c1, c2 = st.columns(2)
        anciennete = c1.slider("Ancienneté (jours depuis l'inscription)", 0, 800, 350)
        recence = c2.slider("Jours depuis le dernier achat", 0, 100, 50)

        st.subheader("Dépenses des 2 dernières années")
        c1, c2, c3 = st.columns(3)
        vin = c1.number_input("Vin", 0, 2000, 174)
        viande = c2.number_input("Viande", 0, 2000, 67)
        poisson = c3.number_input("Poisson", 0, 500, 12)
        c1, c2, c3 = st.columns(3)
        fruits = c1.number_input("Fruits", 0, 500, 8)
        sucre = c2.number_input("Produits sucrés", 0, 500, 8)
        or_ = c3.number_input("Produits « gold »", 0, 500, 24)

        st.subheader("Achats par canal")
        c1, c2, c3, c4, c5 = st.columns(5)
        promos = c1.number_input("Achats en promotion", 0, 30, 2)
        web = c2.number_input("Achats web", 0, 40, 4)
        catalogue = c3.number_input("Achats catalogue", 0, 40, 2)
        magasin = c4.number_input("Achats magasin", 0, 20, 5)
        visites = c5.number_input("Visites web (mois)", 0, 30, 6)

        envoyer = st.form_submit_button("Calculer le score")

    if envoyer:
        ligne = {
            "Income": float(revenu),
            "Kidhome": kidhome,
            "Teenhome": teenhome,
            "Recency": recence,
            "MntWines": vin,
            "MntFruits": fruits,
            "MntMeatProducts": viande,
            "MntFishProducts": poisson,
            "MntSweetProducts": sucre,
            "MntGoldProds": or_,
            "NumDealsPurchases": promos,
            "NumWebPurchases": web,
            "NumCatalogPurchases": catalogue,
            "NumStorePurchases": magasin,
            "NumWebVisitsMonth": visites,
            "AcceptedCmp1": int(cmp1),
            "AcceptedCmp2": int(cmp2),
            "AcceptedCmp3": int(cmp3),
            "AcceptedCmp4": int(cmp4),
            "AcceptedCmp5": int(cmp5),
            "Complain": int(plainte),
            "Age": age,
            "Customer_Seniority": anciennete,
            "Total_Children": kidhome + teenhome,
            "MntTotal": vin + fruits + viande + poisson + sucre + or_,
            "Total_AcceptedCmp": int(cmp1) + int(cmp2) + int(cmp3) + int(cmp4) + int(cmp5),
            "Living_With_Partner": int(couple),
            "Education_Level": niveaux[education],
        }
        X = pd.DataFrame([ligne])[colonnes]
        score = float(modele.predict_proba(X)[0, 1])
        rang = float((scores_base < score).mean() * 100)

        c1, c2 = st.columns(2)
        c1.metric("Score de propension", f"{score:.0%}")
        c2.metric("Situé au-dessus de", f"{rang:.0f} % des clients de la base")
        if score >= 0.5:
            st.success("Au seuil de 50 %, le modèle classe ce client parmi les répondeurs probables.")
        else:
            st.warning("Au seuil de 50 %, le modèle classe ce client parmi les non-répondeurs probables.")
        st.caption(
            "Le score sert surtout à classer les clients entre eux. Le modèle est pondéré pour "
            "compenser le déséquilibre des classes, donc ce n'est pas une probabilité calibrée : "
            "un score de 60 % ne veut pas dire que 6 clients sur 10 de ce profil répondent. "
            "Le rang dans la base est plus fiable à interpréter."
        )
