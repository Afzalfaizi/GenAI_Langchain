from langchain_google_geni import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain.prompts import promptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()



llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))


prompt_template = ChatPromptTemplate.from_message(
    [
        SystemMessage(content="You are Fruits Assistant, you have to act like"),
        SystemMessage(content="please  don't anything else, just  answer related to fruits questions"),
        HumanMessage(content="what is the color of apple?")
    ]
)

prompt = prompt_template.format()

print("Prompt :", prompt)