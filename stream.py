import streamlit as st
import random
import json
from sentence_transformers import SentenceTransformer, util

# Charger les intents depuis un fichier JSON
with open("./base_connaissance.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Charger un modèle de similarité (SentenceTransformer)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Préparer les embeddings pour les patterns
patterns = []
intents_map = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        intents_map.append(intent)

# Créer les embeddings pour les patterns
pattern_embeddings = model.encode(patterns, convert_to_tensor=True)

# Fonction pour identifier l'intent le plus proche
def get_best_intent(user_input: str):
    if not patterns:
        return None
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(user_embedding, pattern_embeddings)
    best_match_idx = similarity_scores.argmax().item()
    best_match_score = similarity_scores[0][best_match_idx].item()
    
    threshold = 0.9  # Seuil de similarité minimal
    if best_match_score < threshold:
        return None
    return intents_map[best_match_idx]

# Fonction principale pour générer une réponse
def generate_response(user_input: str):
    intent = get_best_intent(user_input)
    if intent is None:
        with open("./stock_quest.txt", "a", encoding="utf-8") as file:
            file.write(user_input + "\n")
        return "Je n’ai pas la réponse exacte à cette question, mais pas d’inquiétude ! Vous pouvez contacter le 01 62 45 01 05, et quelqu’un se fera un plaisir de vous aider. Si vous préférez être rappelé à un moment qui vous convient, n’hésitez pas à réserver un créneau sur le site : https://nostrumcare.fr/contact. Ils seront ravis de vous accompagner !"
    return random.choice(intent["responses"])

# Initialisation de l'interface utilisateur
st.title("Chatbot avec Streamlit")
st.write("Bienvenue ! Posez vos questions ci-dessous.")

# Initialiser l'historique des messages
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Affichage des messages sous forme de bulles
for message in st.session_state.conversation_history:
    if message["role"] == "user":
        st.markdown(
            f"""
            <div style='text-align: left; background-color: black; color: white; padding: 10px; 
            border-radius: 10px; margin-bottom: 10px; max-width: 60%;'>
                <b>Vous :</b> {message['content']}
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif message["role"] == "assistant":
        st.markdown(
            f"""
            <div style='text-align: left; background-color: white; color: black; padding: 10px; 
            border-radius: 10px; margin-bottom: 10px; max-width: 60%; margin-left: auto;'>
                <b>Assistant :</b> {message['content']}
            </div>
            """,
            unsafe_allow_html=True,
        )

# Champ de texte pour poser une question
st.write("---")
user_input = st.text_area(
    "Votre message",
    placeholder="Écrivez ici votre message...",
    label_visibility="collapsed",
    height=68,
)

if st.button("Envoyer"):
    if not user_input.strip():
        st.warning("Veuillez entrer un message avant d'envoyer.")
    else:
        bot_response = generate_response(user_input)
        
        # Stocker l'historique de la conversation
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        st.session_state.conversation_history.append({"role": "assistant", "content": bot_response})
        
        st.rerun()
