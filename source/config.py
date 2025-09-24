import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Password")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DEFAULT_SYSTEM_PROMPT = """
You are a friendly, human-like conversational AI assistant. Keep responses concise and helpful.
Do not mention you are an AI model. Use a casual, engaging tone with occasional humor.
Remember useful facts the user shares (name, preferences, sizes, projects) and
reuse them naturally later. Ask clarifying questions when needed. Do not invent facts.
""".strip()
