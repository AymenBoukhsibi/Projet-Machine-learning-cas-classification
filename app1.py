import streamlit as st
import pandas as pd
import pickle

# --- CONFIGURATION DE L'INTERFACE ---
st.write('''
# Application de Prédiction du Risque de Crédit
Cette interface utilise un modèle **Random Forest** pour évaluer le risque client.
''')

st.sidebar.header("Paramètres d'entrée du Client :")

# --- 1. COLLECTE DES DONNÉES (SLIDERS) ---
def user_input():
    # Basé sur vos colonnes A1 à A7
    a1 = st.sidebar.slider('Variable A1 (ex: Age/Revenu)', 0.0, 100.0, 25.0)
    a2 = st.sidebar.slider('Variable A2', 0.0, 30.0, 5.0)
    a3 = st.sidebar.slider('Variable A3', 0.0, 1000.0, 200.0)
    a4 = st.sidebar.slider('Variable A4', 0.0, 1000.0, 100.0)
    
    # Pour les variables catégorielles (A5, A6, A7), on utilise des selectbox
    a5 = st.sidebar.selectbox('Variable A5 (g/p)', [0, 1]) # Converti en numérique pour le modèle
    a6 = st.sidebar.selectbox('Variable A6', [0, 1, 2])
    a7 = st.sidebar.selectbox('Variable A7', [0, 1, 2])
    
    data = {
        'A1': a1, 'A2': a2, 'A3': a3, 'A4': a4,
        'A5': a5, 'A6': a6, 'A7': a7
    }
    return pd.DataFrame(data, index=[0])

# Stockage des données saisies
df = user_input()

# --- 2. CHARGEMENT DU MODÈLE ---
# Utilisation du nom de fichier validé dans vos dernières étapes
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    # --- 3. CALCULS ET PRÉDICTIONS ---
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
    
    # --- 4. AFFICHAGE DES TROIS ÉLÉMENTS ---
    
    # Élément 1 : Tableau des données saisies
    st.subheader("1. Données du profil à analyser")
    st.write(df)

    # Élément 2 : Résultat de la Prédiction
    st.subheader("2. Résultat de la Classification")
    # Correspond aux labels de votre dataset
    labels = ["Risque Élevé", "Risque Faible"] 
    resultat = labels[prediction[0]]
    
    if prediction[0] == 1:
        st.success(f"Le modèle prédit un : **{resultat}** ✅")
    else:
        st.error(f"Le modèle prédit un : **{resultat}** ❌")

    # Élément 3 : Détails des Probabilités
    st.subheader("3. Probabilités de certitude")
    df_proba = pd.DataFrame(prediction_proba, columns=labels)
    st.write(df_proba)

    # Rappel de la performance
    st.info(f"Précision du modèle entraîné : 78.67%")

except FileNotFoundError:
    st.error("⚠️ Erreur : Le fichier 'model.pkl' est introuvable dans le répertoire actuel.")