from langchain_groq import ChatGroq
from src.config import GROQ_API_KEY, GROQ_MODEL


def get_llm(temperature: float = 0.7) -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=temperature,
    )
