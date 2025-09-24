from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_EPISODE_MENTIONS


class GraphitiService:
    def __init__(self, client: Graphiti):
        self.client = client

    @classmethod
    async def create(cls, uri: str, user: str, password: str):
        client = Graphiti(uri, user, password)
        await client.build_indices_and_constraints()
        return cls(client)

    async def get_or_create_user_uuid(self, user_name: str) -> str:
        nl = await self.client._search(user_name, NODE_HYBRID_SEARCH_EPISODE_MENTIONS)
        if getattr(nl, "nodes", []):
            return nl.nodes[0].uuid

        # Create a seed episode to anchor the user node
        await self.client.add_episode(
            name="User Creation",
            episode_body=f"{user_name} started a chat.",
            source=EpisodeType.text,
            reference_time=datetime.now(timezone.utc),
            source_description="GradioChat",
        )
        nl = await self.client._search(user_name, NODE_HYBRID_SEARCH_EPISODE_MENTIONS)
        return nl.nodes[0].uuid

    async def search_user_facts(self, center_uuid: str, query: str, num_results: int = 8) -> list[EntityEdge]:
        edges = await self.client.search(
            query,
            center_node_uuid=center_uuid,
            num_results=num_results,
        )
        return edges

    async def log_exchange(self, user_name: str, user_message: str, assistant_message: str) -> None:
        await self.client.add_episode(
            name="Chatbot Response",
            episode_body=f"{user_name}: {user_message}\nAI friend: {assistant_message}",
            source=EpisodeType.message,
            reference_time=datetime.now(timezone.utc),
            source_description="Chatbot",
        )

    async def clear_all_data(self) -> None:
        """Deletes all nodes and relationships from the graph."""
        # This is a raw Cypher query to delete everything.
        # The graphiti client is assumed to expose the neo4j driver.
        if hasattr(self.client, "driver"):
            await self.client.driver.execute_query("MATCH (n) DETACH DELETE n")
        await self.client.build_indices_and_constraints()