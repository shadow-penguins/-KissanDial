import requests

# Sarvam AI API configuration
SARVAM_API_URL = "https://api.sarvam.ai/v1/translate"
SARVAM_API_KEY = "9afb46fb-f0ca-4565-b7ea-e2c06c4d7fc4"

def translate_text(text: str, target_language: str) -> str:
    """
    Translate text to the target language using Sarvam AI API.
    """
    headers = {
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "target_language": target_language
    }
    
    response = requests.post(SARVAM_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json().get("translated_text", text)
    else:
        print(f"Translation error: {response.text}")
        return text  # Fallback to original text on error

@app.route("/handle-speech", methods=['POST'])
async def handle_speech():
    global to_say
    resp = VoiceResponse()

    speech_result = request.form.get('SpeechResult')
    target_language = request.form.get('Language', 'en')  # Default to English if no language is provided
    
    if speech_result:
        # Process the speech result through the agent
        agent_response = agent.chat(speech_result)
        print(f":User  {speech_result}")
        print(f"Assistant: {agent_response}")
        
        # Translate the agent's response if needed
        translated_response = translate_text(str(agent_response), target_language)
        
        to_say = translated_response
        resp.redirect("/voice")
    else:
        resp.say("I'm sorry, I didn't catch that. Could you please repeat?")
        resp.redirect("/voice")
    return str(resp)