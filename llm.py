from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 
load_dotenv()

def create_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="openai/gpt-oss-120b")
    return llm
