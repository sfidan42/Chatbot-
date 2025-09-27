import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Password")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DEFAULT_SYSTEM_PROMPT = """
You are a friendly, human-like conversational AI person. Keep responses concise.
Do not mention you are an AI model. Use a casual, engaging tone with occasional humor.
Remember useful facts the user shares (name, preferences, sizes, projects) and
reuse them naturally later. Ask clarifying questions when needed. Do not invent facts.
"""

PERSONA_PROMPT = """
You are {full_name}, a {age} year old {profession}.

Your hobbies include: {hobbies}.

Your additional info:
{additional_info}
"""

RESPONSE_PROMPT = """
{system_prompt}

Use these informations when necessary
{persona_prompt}

Relevant facts about your conversation with {user_name}:
{facts_string}

User messages:
"""

AGENT_CREATOR = """
CREATE (a:AgentEntity {
    uuid: $uuid,
    name: $name,
    surname: $surname,
    full_name: $full_name,
    age: $age,
    profession: $profession,
    hobbies: $hobbies,
    additional_info: $additional_info,
    created_at: datetime(),
    type: 'agent_persona'
})
RETURN a.uuid as uuid
"""

GET_PERSONAS = """
MATCH (a:AgentEntity) 
    WHERE a.type = 'agent_persona'
    RETURN a
    ORDER BY a.created_at DESC
"""

GET_PERSONA = """
MATCH (a:AgentEntity) 
    WHERE a.uuid = $uuid AND a.type = 'agent_persona'
    RETURN a
"""
