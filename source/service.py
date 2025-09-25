from datetime import datetime, timezone
import json
import uuid

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType


class GraphitiService:
    def __init__(self, client: Graphiti):
        self.client = client

    @classmethod
    async def create(cls, uri: str, user: str, password: str):
        client = Graphiti(uri, user, password)
        await client.build_indices_and_constraints()
        return cls(client)

    async def get_or_create_user_uuid(self, user_name: str) -> str:
        # Search for existing user
        query = f"MATCH (n:Entity) WHERE n.name = '{user_name}' RETURN n LIMIT 1"
        async with self.client.driver.session() as session:
            result = await session.run(query)
            records = await result.data()
            if records:
                return records[0]['n']['uuid']

        # Create a seed episode to anchor the user node
        await self.client.add_episode(
            name="User Creation",
            episode_body=f"{user_name} started a chat.",
            source=EpisodeType.text,
            reference_time=datetime.now(timezone.utc),
            source_description="GradioChat",
        )
        
        # Search again for the created user
        async with self.client.driver.session() as session:
            result = await session.run(query)
            records = await result.data()
            if records:
                return records[0]['n']['uuid']
        return ""

    async def create_agent_persona(self, name: str, surname: str, age: int, profession: str, hobbies: str, additional_info: str = "") -> str:
        """Create an agent persona as AgentEntity and return its UUID"""
        try:
            full_name = f"{name} {surname}"
            persona_uuid = f"agent_{uuid.uuid4().hex[:8]}"
            
            # Create AgentEntity node directly
            query = """
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
            
            async with self.client.driver.session() as session:
                result = await session.run(query, {
                    'uuid': persona_uuid,
                    'name': name,
                    'surname': surname,
                    'full_name': full_name,
                    'age': age,
                    'profession': profession,
                    'hobbies': hobbies,
                    'additional_info': additional_info
                })
                records = await result.data()
                if records:
                    return records[0]['uuid']
            return ""
        except Exception as e:
            print(f"Error creating agent persona: {e}")
            return ""

    async def get_all_personas(self) -> list[dict]:
        """Get all existing agent personas"""
        try:
            query = """
            MATCH (a:AgentEntity) 
            WHERE a.type = 'agent_persona'
            RETURN a
            ORDER BY a.created_at DESC
            """
            
            async with self.client.driver.session() as session:
                result = await session.run(query)
                records = await result.data()
                
                personas = []
                for record in records:
                    node = record['a']
                    personas.append({
                        'uuid': node['uuid'],
                        'name': node['name'],
                        'surname': node['surname'],
                        'full_name': node['full_name'],
                        'age': str(node['age']),
                        'profession': node['profession'],
                        'hobbies': node['hobbies'],
                        'additional_info': node.get('additional_info', '')
                    })
                
                return personas
        except Exception as e:
            print(f"Error getting personas: {e}")
            return []

    async def get_persona_by_uuid(self, uuid: str) -> dict:
        """Get persona details by UUID"""
        try:
            query = """
            MATCH (a:AgentEntity) 
            WHERE a.uuid = $uuid AND a.type = 'agent_persona'
            RETURN a
            """
            
            async with self.client.driver.session() as session:
                result = await session.run(query, {'uuid': uuid})
                records = await result.data()
                
                if records:
                    node = records[0]['a']
                    return {
                        'uuid': node['uuid'],
                        'name': node['name'],
                        'surname': node['surname'],
                        'full_name': node['full_name'],
                        'age': str(node['age']),
                        'profession': node['profession'],
                        'hobbies': node['hobbies'],
                        'additional_info': node.get('additional_info', '')
                    }
            return {}
        except Exception as e:
            print(f"Error getting persona: {e}")
            return {}

    async def search_user_facts(self, center_uuid: str, query: str, num_results: int = 8) -> list[EntityEdge]:
        edges = await self.client.search(
            query,
            center_node_uuid=center_uuid,
            num_results=num_results,
        )
        return edges

    async def log_exchange(self, user_name: str, user_message: str, assistant_message: str, ai_name: str = "AI friend") -> None:
        # Include both user name and AI name to differentiate conversations
        await self.client.add_episode(
            name=f"Chat: {user_name} with {ai_name}",
            episode_body=f"User {user_name}: {user_message}\n{ai_name}: {assistant_message}",
            source=EpisodeType.message,
            reference_time=datetime.now(timezone.utc),
            source_description=f"Chatbot-{ai_name}",
        )
