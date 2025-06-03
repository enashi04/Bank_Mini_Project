import streamlit as st
import numpy as np
import pickle #pour le chargement du modèle et du scaler

# Chargement du modèle et du scaler
knn = pickle.load(open('knn_model_app.pkl', 'rb'))
scaler = pickle.load(open('scaler_app.pkl', 'rb'))

# Configuration de la page
st.set_page_config(page_title="Souscription Bancaire", page_icon="banque.png", layout="centered")

#titre 
st.title("💡 Prédiction de Souscription à une Offre Bancaire")
#description
st.markdown("Utilisez ce simulateur pour estimer si un client est susceptible de souscrire à une offre de **dépôt à terme**. Renseignez les caractéristiques ci-dessous puis cliquez sur *Prédire*.")

st.markdown("---")

# Saisie utilisateur
st.subheader("📋 Informations client")
col1, col2 = st.columns(2) #on divise en deux colonnes (comme un flex en css)

with col1: #colonne un (gauche)
    age = st.slider('Âge', 18, 95, 35) #barre d'age
    balance = st.number_input('💰 Solde moyen annuel (€)', value=500) #text pour mettre le solde

with col2: #colonne deux (droite)
    campaign = st.slider("☎️ Nombre de contacts durant la campagne", 1, 50, 1) #barre de nbre de contact
    pdays = st.slider("📆 Jours depuis la dernière campagne", -1, 999, -1) #barre de jour de derniere campagne
    previous = st.slider("📊 Nombre de contacts précédents", 0, 50, 0)# barre de contact précédent

# Données utilisateur
input_data = np.array([[age, balance, campaign, pdays, previous]]) #entrée donnée user
input_scaled = scaler.transform(input_data) #transformation

st.markdown("---")

# Bouton de prédiction
if st.button("🔍 Prédire"):
    prediction = knn.predict(input_scaled) #prédiction avec knn
    probability = knn.predict_proba(input_scaled)[0][1] * 100 #probabilité

    if prediction[0] == 1: #veut dire que c'est ok
        st.success("✅ Ce client est **susceptible de souscrire** à l’offre.")
        st.markdown(f"🔢 Probabilité estimée : **{probability:.2f}%**")
    else: #pas ok
        st.warning("❌ Ce client n’est **probablement pas intéressé** par l’offre.")
        st.markdown(f"🔢 Probabilité estimée : **{probability:.2f}%**")

# Crédit
st.markdown("---")
st.caption("👩‍💻 Projet réalisé par Ihsane ERRAMI· M1 Data Management· 2025")

