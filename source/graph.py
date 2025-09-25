from typing import Annotated

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, add_messages, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from graphiti_core.edges import EntityEdge


def edges_to_facts_string(entities: list[EntityEdge]) -> str:
    if not entities:
        return ""
    return "-" + "\n- ".join([edge.fact for edge in entities])


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_name: str
    user_node_uuid: str
    system_prompt: str
    ai_persona: dict


class AgentRunner:
    def __init__(self, service, model: str, temperature: float = 0.5):
        self.service = service
        self.llm = ChatOllama(model=model, temperature=temperature)
        self.graph_builder = StateGraph(AgentState)
        self.memory = MemorySaver()

        self.graph_builder.add_node("agent", self.chatbot)
        self.graph_builder.add_edge(START, "agent")
        self.graph_builder.add_edge("agent", END)

        self.graph = self.graph_builder.compile(checkpointer=self.memory)

    def _build_persona_prompt(self, persona: dict) -> str:
        """Build persona-specific prompt"""
        if not persona:
            return ""
        
        name = persona.get('name', 'AI')
        full_name = persona.get('full_name', name)
        
        persona_prompt = f"""
You are {full_name}, a {persona.get('age', '')} year old {persona.get('profession', '')}.
Your hobbies include: {persona.get('hobbies', 'various activities')}.
{persona.get('additional_info', '')}

IMPORTANT: You are {full_name}, not any other person mentioned in the conversation history.
When responding, always remember you are {name} and the user is the person you're currently talking to.
Do not confuse the user with other people mentioned in past conversations.

Remember to stay in character and respond as {name} would.
When asked about your name or identity, you are {full_name}.
"""
        return persona_prompt.strip()

    async def chatbot(self, state: AgentState):
        user_name = state["user_name"]
        user_uuid = state["user_node_uuid"]
        system_prompt = (state.get("system_prompt") or "").strip()
        persona = state.get("ai_persona", {})

        last_user_msg = ""
        for m in reversed(state["messages"]):
            role = getattr(m, "role", getattr(m, "type", ""))
            if role in ("user", "human"):
                last_user_msg = getattr(m, "content", "")
                break

        facts = await self.service.search_user_facts(
            user_uuid,
            f"{user_name}: {last_user_msg}",
            num_results=8,
        )
        facts_string = edges_to_facts_string(facts) or "No facts."

        # Build combined prompt with persona info
        persona_prompt = self._build_persona_prompt(persona)
        combined_prompt = f"{system_prompt}\n\n{persona_prompt}" if persona_prompt else system_prompt

        system_message = SystemMessage(
            content=f"""{combined_prompt}

Facts about the user and their conversation so far:
{facts_string}"""
        )

        messages = [system_message] + state["messages"]
        response = await self.llm.ainvoke(messages)

        # Get AI name from persona
        ai_name = persona.get("full_name", "AI friend") if persona else "AI friend"

        # Persist the exchange
        try:
            import asyncio
            asyncio.create_task(self.service.log_exchange(user_name, last_user_msg, response.content, ai_name))
        except Exception:
            await self.service.log_exchange(user_name, last_user_msg, response.content, ai_name)

        return {"messages": [response]}
    
    async def astream_response(self, state, thread_id: str, ai_name: str = "AI friend"):
        """Stream the response token by token"""
        user_name = state["user_name"]
        user_uuid = state["user_node_uuid"]
        system_prompt = (state.get("system_prompt") or "").strip()
        persona = state.get("ai_persona", {})

        last_user_msg = ""
        for m in reversed(state["messages"]):
            role = getattr(m, "role", getattr(m, "type", ""))
            if role in ("user", "human"):
                last_user_msg = getattr(m, "content", "")
                break

        # Search for facts but be more specific about the context
        persona_name = persona.get('full_name', persona.get('name', 'AI'))
        facts = await self.service.search_user_facts(
            user_uuid,
            f"Current user {user_name} talking to {persona_name}: {last_user_msg}",
            num_results=6,
        )
        facts_string = edges_to_facts_string(facts) or "No relevant facts found."

        # Build combined prompt with persona info
        persona_prompt = self._build_persona_prompt(persona)
        combined_prompt = f"{system_prompt}\n\n{persona_prompt}" if persona_prompt else system_prompt

        system_message = SystemMessage(
            content=f"""{combined_prompt}

Current conversation context:
- You are speaking with: {user_name}
- You are: {persona_name}

Relevant facts about your conversation with {user_name}:
{facts_string}

Remember: You are {persona_name}, and you are currently talking to {user_name}. Do not confuse them with other people from your memory."""
        )

        messages = [system_message] + state["messages"]
        
        # Stream the response
        full_response = ""
        async for chunk in self.llm.astream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                full_response += chunk.content
                yield full_response

        # Persist the exchange after streaming is complete
        try:
            import asyncio
            asyncio.create_task(self.service.log_exchange(user_name, last_user_msg, full_response, ai_name))
        except Exception:
            await self.service.log_exchange(user_name, last_user_msg, full_response, ai_name)

    async def ainvoke(self, state, thread_id: str):
        return await self.graph.ainvoke(state, config={"configurable": {"thread_id": thread_id}})