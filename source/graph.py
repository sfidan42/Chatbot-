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

    async def chatbot(self, state: AgentState):
        user_name = state["user_name"]
        user_uuid = state["user_node_uuid"]
        system_prompt = (state.get("system_prompt") or "").strip()

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

        system_message = SystemMessage(
            content=f"""{system_prompt}

Facts about the user and their conversation so far:
{facts_string}"""
        )

        messages = [system_message] + state["messages"]
        response = await self.llm.ainvoke(messages)

        # Persist the exchange
        try:
            import asyncio
            asyncio.create_task(self.service.log_exchange(user_name, last_user_msg, response.content))
        except Exception:
            await self.service.log_exchange(user_name, last_user_msg, response.content)

        return {"messages": [response]}
    
    async def astream_response(self, state, thread_id: str):
        """Stream the response token by token"""
        user_name = state["user_name"]
        user_uuid = state["user_node_uuid"]
        system_prompt = (state.get("system_prompt") or "").strip()

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

        system_message = SystemMessage(
            content=f"""{system_prompt}

Facts about the user and their conversation so far:
{facts_string}"""
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
            asyncio.create_task(self.service.log_exchange(user_name, last_user_msg, full_response))
        except Exception:
            await self.service.log_exchange(user_name, last_user_msg, full_response)


    async def ainvoke(self, state, thread_id: str):
        return await self.graph.ainvoke(state, config={"configurable": {"thread_id": thread_id}})
