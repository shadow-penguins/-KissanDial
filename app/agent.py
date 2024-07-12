from flask import Flask, request
from twilio.twiml.voice_response import VoiceResponse, Gather
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.memory import ChatMemoryBuffer
import pandas as pd
from twilio.rest import Client


# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

to_say = 'Hi welcome to the Agricultural Assistant. How can I help you today?'

app = Flask(__name__)

# Load and index documents
subsidy_docs = SimpleDirectoryReader(input_files=["main_subsidy_data.csv"]).load_data()
subsidy_index = VectorStoreIndex.from_documents(subsidy_docs)

# Create query engine
subsidy_engine = subsidy_index.as_query_engine(similarity_top_k=6)

# Load the CSV file
df = pd.read_csv("main_subsidy_data.csv")

def send_sms_with_subsidy_info(query: str) -> str:
    """
    Send SMS with 'how_to_apply' information for relevant subsidies.
    """
    print("INSIDE SEND SMS FUNCTION")
    print(f"Query: {query}")
    results = subsidy_engine.query(query)
    print(f"Results: {results}")
    
    sms_body = "How to apply for relevant subsidies:\n\n"
    # for subsidy in relevant_subsidies:
    sms_body += f"{results}"

    account_sid = ''
    auth_token = ''
    client = Client(account_sid, auth_token)
    
    try:
        message = client.messages.create(
            from_='+17206276852',
            body=sms_body,
            to="+919108006252"
        )
        return f"SMS sent successfully with SID: {message.sid}"
    except Exception as e:
        return f"Error sending SMS: {str(e)}"

# Define tools
subsidy_tool = QueryEngineTool(
    query_engine=subsidy_engine,
    metadata=ToolMetadata(
        name="get_subsidy_info",
        description="Provides information about subsidies for farmers in India.",
    ),
)

sms_tool = FunctionTool.from_defaults(
    fn=send_sms_with_subsidy_info,
    name="send_sms_with_subsidy_info",
    description="Sends an SMS with 'how_to_apply' information for relevant subsidies.",
)

# Custom prompt for the agent
CUSTOM_PROMPT = """
You are a helpful conversational assistant. Below are the global details about the usecase which you need to abide by strictly:
<global_details>
Task: You are an intelligent agricultural assistant. Your goal is to provide helpful information to farmers about subsidies, weather, and farming advice. Always ask follow-up questions to understand the farmer's specific needs better before giving detailed information.
Use the provided tools to gather information, but always ask follow-up questions before providing detailed answers. This will help you give more targeted and useful information to the farmer. For eg: "Agent: Are you looking for subsidies related to seeds, crops, machines or insurance?"
Response style: Your responses must be very short, concise and helpful.
</global_details>

You are currently in a specific state of a conversational flow which is described below. This state is one amongst many other states which constitutes the entire conversation design of the interaction between a user and you as the assistant. Details about the current state:
<state_details>
Name: Help farmer
Goal: To answer all queries regarding government subsidies for farmers and weather.
Instructions: 1. Greet the user and understand their question. 2. If the user asks about subsidies always ask them if they are looking for subsidies for crops, seeds, machines or insurance if this information is not already known. If the user asks about the weather, call TOOL: get_weather_info 3. Answer user's questions using fetched data.
Tools: ["get_subsidy_info", "get_weather_info", "send_sms_with_subsidy_info"]
</state_details>

Remember to follow the below rules strictly:
1. Ensure coherence in the conversation. Responses should engage the user and maintain a flow that feels like a natural dialogue.
2. Use only the available tools in the bot's response.
3. Avoid generating any Unicode or Hindi characters.
4. Do not address the user and refrain from using any name for the user.
5. Strictly refrain from using emojis in your responses.
6. Your responses should be engaging, short and crisp. It should be more human conversation like.

- Use informal, more conversational and colloquial language. Avoid decorative words and choice of too much drama in your language.
- Avoid bulleted lists, markdowns, structured outputs, and any special characters like double quotes, single quotes, or hyphen etc in your responses.
- Avoid any numericals or SI units in your outputs. Ex: if you want to say 12:25, say twelve twenty five, or if you want to say 100m^2, say hundred meter square since this interaction is over call. Other fields can have numericals.
- Avoid any emoji or smiley faces since this interaction is over call.
- Call relevant tools whether it be some api or a retrieval tool to fetch context needed to answer any query that the user might have. First decide if a tool call is needed in the thought and then call the appropriate tool. Respond to the user with a variant of 'let me check that for you' and then call the tool in the same turn.
"""

# Initialize the agent
llm = OpenAI(temperature=0, model="gpt-4o")
memory = ChatMemoryBuffer.from_defaults(token_limit=2048)
llm = OpenAI(temperature=0, model="gpt-4")
memory = ChatMemoryBuffer.from_defaults(token_limit=2048)
agent_worker = FunctionCallingAgentWorker.from_tools(
  [subsidy_tool, sms_tool], system_prompt=CUSTOM_PROMPT, 
  memory=memory, llm=llm
  verbose=True,
  allow_parallel_tool_calls=False,
)
agent = agent_worker.as_agent()

@app.route("/voice", methods=['POST'])
def voice():
    global to_say
    print(f"to_say: {to_say}")
    resp = VoiceResponse()
    
    gather = Gather(
        input="speech",
        action="/handle-speech",
        method="POST",
        speechTimeout="1",
        speechModel="experimental_conversations",
        enhanced=True
    )
    gather.say(to_say)
    resp.append(gather)
    
    resp.redirect("/voice")
    
    return str(resp)

@app.route("/handle-speech", methods=['POST'])
async def handle_speech():
    global to_say
    resp = VoiceResponse()

    speech_result = request.form.get('SpeechResult')
    
    if speech_result:
        agent_response = agent.chat(speech_result)
        print(f"User: {speech_result}")
        print(f"Assistant: {agent_response}")
        to_say = str(agent_response)
        resp.redirect("/voice")
    else:
        resp.say("I'm sorry, I didn't catch that. Could you please repeat?")
        resp.redirect("/voice")
    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)
