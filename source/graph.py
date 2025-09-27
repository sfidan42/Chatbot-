from typing import Annotated

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, add_messages, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from graphiti_core.edges import EntityEdge
from source.config import PERSONA_PROMPT, RESPONSE_PROMPT

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_name: str
    user_node_uuid: str
    system_prompt: str
    ai_persona: dict


class AgentRunner:
    def __init__(self, service, model: str, temperature: float = 0.5):
        self.service = service
        self.llm = ChatOpenAI(model=model, temperature=temperature) 
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
        
        persona_prompt = PERSONA_PROMPT.format(
            full_name=full_name,
            age=persona.get('age', ''),
            profession=persona.get('profession', ''),
            hobbies=persona.get('hobbies', 'various activities'),
            additional_info=persona.get('additional_info', ''),
        ).strip()
        return persona_prompt

    async def chatbot(self, state: AgentState):
        user_name = state["user_name"]
        user_uuid = state["user_node_uuid"]
        system_prompt = (state.get("system_prompt") or "").strip()
        persona = state.get("ai_persona", {})

        last_user_msg = ""
        for m in reversed(state["messages"]):
            role = m.get("role", m.get("type", "")) 
            if role in ("user", "human"):
                last_user_msg = m.get("content", "")
                break


        facts_string = await self.service.search_user_facts(
            user_uuid,
            f"{user_name}: {last_user_msg}",
            num_results=8,
        )

        persona_prompt = self._build_persona_prompt(persona)

        response_prompt = RESPONSE_PROMPT.format(system_prompt, persona_prompt, user_name, facts_string)

        system_message = SystemMessage(content=response_prompt)

        messages = [system_message] + state["messages"]
        response = await self.llm.ainvoke(messages)

        ai_name = persona.get("full_name", "AI friend") if persona else "AI friend"

        try:
            import asyncio
            asyncio.create_task(self.service.log_exchange(user_name, last_user_msg, response.content, ai_name))
        except Exception:
            await self.service.log_exchange(user_name, last_user_msg, response.content, ai_name)

        return {"messages": [response]}
    
    async def astream_response(self, state, thread_id: str, ai_name: str = "AI friend"):
        user_name = state["user_name"]
        user_uuid = state["user_node_uuid"]
        persona = state.get("ai_persona", {})
        system_prompt=(state.get("system_prompt") or "").strip()

        last_user_msg = ""
        for m in reversed(state["messages"]):
            role = m.get("role", m.get("type", "")) 
            if role in ("user", "human"):
                last_user_msg = m.get("content", "")
                break
        
        # Search for facts but be more specific about the context
        persona_name = persona.get('full_name', persona.get('name', 'AI'))
        persona_prompt = self._build_persona_prompt(persona)

        facts_string = await self.service.retrieve_informations(
            user_uuid,
            f"Current user {user_name} talking to {persona_name}: {last_user_msg}",
            num_results=6,
        )

        response_prompt = RESPONSE_PROMPT.format(
            system_prompt=system_prompt,
            persona_prompt=persona_prompt,
            user_name=user_name,
            facts_string=facts_string
        )

        system_message = SystemMessage(content=response_prompt)

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
