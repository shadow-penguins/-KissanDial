from flask import Flask, request, jsonify
from twilio.twiml.voice_response import VoiceResponse, Gather
import os
import requests
import pandas as pd
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

# Sarvam AI API Key
SARVAM_AI_API_KEY = "9afb46fb-f0ca-4565-b7ea-e2c06c4d7fc4"

# Flask app
app = Flask(__name__)

# Load and index documents
subsidy_docs = SimpleDirectoryReader(input_files=["main_subsidy_data.csv"]).load_data()
subsidy_index = VectorStoreIndex.from_documents(subsidy_docs)
subsidy_engine = subsidy_index.as_query_engine(similarity_top_k=6)

# Load the CSV file
df = pd.read_csv("main_subsidy_data.csv")

# Sarvam AI translation function
def translate_text(text, target_language):
    """Translate text using Sarvam AI."""
    url = "https://api.sarvam.ai/translate"
    headers = {"Authorization": f"Bearer {SARVAM_AI_API_KEY}"}
    data = {
        "source_language": "auto",  # Detect source language automatically
        "target_language": target_language,
        "text": text
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json().get("translated_text")
    else:
        return None

# Language detection function (using Sarvam AI)
def detect_language(text):
    """Detect the language of the input text."""
    url = "https://api.sarvam.ai/detect_language"
    headers = {"Authorization": f"Bearer {SARVAM_AI_API_KEY}"}
    data = {"text": text}
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json().get("language")
    else:
        return None

# Route for voice interaction
@app.route("/voice", methods=["POST"])
def voice():
    """Handle incoming voice calls and respond with appropriate subsidy information."""
    # Extract user's speech input
    user_input = request.form.get("SpeechResult")
    
    # Detect the language of the user's input
    detected_language = detect_language(user_input)
    
    # If input is not in English, translate it
    if detected_language != "en":
        user_input = translate_text(user_input, "en")
    
    # Query the subsidy engine
    results = subsidy_engine.query(user_input)
    
    # Convert the results into the original language (if necessary)
    if detected_language != "en":
        results = translate_text(results, detected_language)
    
    # Create a voice response
    response = VoiceResponse()
    response.say(results)
    
    return str(response)

# Route for handling SMS queries
@app.route("/sms", methods=["POST"])
def sms():
    """Handle incoming SMS messages and reply with subsidy information."""
    user_input = request.form.get("Body")
    
    # Detect the language of the user's input
    detected_language = detect_language(user_input)
    
    # If input is not in English, translate it
    if detected_language != "en":
        user_input = translate_text(user_input, "en")
    
    # Query the subsidy engine
    results = subsidy_engine.query(user_input)
    
    # Translate the results back to the original language (if necessary)
    if detected_language != "en":
        results = translate_text(results, detected_language)
    
    # Send SMS response
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=results,
        from_="+1234567890",  # Your Twilio number
        to=request.form.get("From")  # User's number
    )
    
    return jsonify({"message": "SMS sent successfully."})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
