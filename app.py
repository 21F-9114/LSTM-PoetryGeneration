import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load trained model
model = tf.keras.models.load_model("lstm_poetry_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Function to generate poetry
def generate_poetry(seed_text, max_words=50):
    poetry_output = seed_text
    
    for _ in range(max_words):
        # Convert seed text to tokens
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        # Check if token_list is empty or too short
        if len(token_list) == 0:
            poetry_output = "No valid tokens found in the seed text. Please enter a valid phrase."
            break
        
        # Reshape for model input (add batch dimension)
        token_list = np.array(token_list).reshape(1, -1)
        
        # Ensure model receives valid input (pad if necessary)
        # You may need to adjust this part to match your tokenizer settings
        token_list = tf.keras.preprocessing.sequence.pad_sequences(token_list, padding='pre', maxlen=100)
        
        # Make the prediction
        prediction = model.predict(token_list)
        
        # Get predicted word index and map it to the word
        predicted = np.argmax(prediction, axis=-1)[0]
        output_word = tokenizer.index_word.get(predicted, "")
        
        # If the predicted word is empty, break the loop
        if output_word == "":
            break
        
        # Append the predicted word to the seed text for the next prediction
        seed_text += " " + output_word
        poetry_output = seed_text
    
    return poetry_output

# Streamlit UI Configuration
st.set_page_config(page_title="Rang-e-Sukhan - Poetry Generator", layout="centered")

# Custom CSS for a Beautiful UI
st.markdown("""
    <style>
        /* Background Image */
        body {
            background-image: url('https://source.unsplash.com/1600x900/?moon,night,stars');
            background-size: cover;
            background-position: center;
            color: #8E2DE2 → #4A00E0;
        }
        
        /* Title Styling */
        .title {
            text-align: center;
            font-size: 45px;
            font-weight: bold;
            font-family: 'Georgia', serif;
            background: linear-gradient(90deg, #FFD700, #FF5733, #C70039);
            -webkit-background-clip: text;
            color: transparent;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color:   #FF7518 ;
            font-style: italic;
            margin-bottom: 20px;
        }

        /* Input Box */
        .stTextInput>div>div>input {
            border-radius: 10px;
            border: 2px solid #FF4500 ;
            background-color: #1A1A1A;
            color: white;
            padding: 14px;
            font-size: 18px;
            text-align: center;
            box-shadow: 0px 0px 15px rgba(255, 215, 0, 0.7);
        }

        /* Gradient Button */
        .stButton>button {
            background: linear-gradient(90deg, #FF5733, #FF007F);
            color: white;
            font-size: 20px;
            font-weight: bold;
            border-radius: 15px;
            padding: 12px 25px;
            border: none;
            transition: 0.4s;
            box-shadow: 0px 4px 15px rgba(255, 87, 51, 0.6);
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #FF4500, #FF1493);
            transform: scale(1.08);
        }

        /* Poetry Display Box */
        .poetry-box {
            background: rgba(30, 30, 30, 0.8);
            color: white;
            padding: 20px;
            border-radius: 15px;
            font-size: 20px;
            font-family: 'Georgia', serif;
            text-align: center;
            margin-top: 20px;
            box-shadow: 0px 0px 20px rgba(255, 165, 0, 0.6);
            animation: fadeIn 1.5s ease-in-out;
        }

        /* Download Button */
        .stDownloadButton>button {
            background: linear-gradient(90deg, #4CAF50, #00C853);
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 12px;
            padding: 12px 20px;
            border: none;
            box-shadow: 0px 4px 10px rgba(76, 175, 80, 0.6);
            transition: 0.4s;
        }
        .stDownloadButton>button:hover {
            background: linear-gradient(90deg, #388E3C, #00796B);
            transform: scale(1.05);
        }

        /* Animations */
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(10px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        /* Footer */
        .footer {
            text-align: center;
            font-size: 14px;
            margin-top: 30px;
            color: #DDDDDD;
            font-style: italic;
        }
    </style>
""", unsafe_allow_html=True)

# App Title with Elegant Gradient Effect
st.markdown("<h1 class='title'>🌙 Rang-e-Sukhan 🎶</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>A Roman Urdu AI Poetry Generator. Enter a phrase and witness the magic of words! ✨</p>", unsafe_allow_html=True)

# User Input
seed_text = st.text_input("Start your poetry:", "Write here...")

# Poetry variable
poetry = ""

# Poetry Generation Button
if st.button("💖 Generate Poetry"):
    with st.spinner("Crafting poetic beauty... 🎭"):
        poetry = generate_poetry(seed_text)
    
    st.toast("Your poetry is ready! 🌟", icon="💖")
    st.success("Here is your AI-generated poetry:")

    # Display Generated Poetry
    st.markdown(f"<p class='poetry-box'>{poetry}</p>", unsafe_allow_html=True)
    
    # Celebration Animation
    st.snow()
    
    # Download Button
    st.download_button("📜 Save Poetry", poetry, file_name="generated_poetry.txt", key="poetry_download")

# Footer
st.markdown("<p class='footer'>Rang e Sukhan ✨</p>", unsafe_allow_html=True)
