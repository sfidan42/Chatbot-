import uuid
import gradio as gr

from source.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    DEFAULT_SYSTEM_PROMPT,
)
from source.service import GraphitiService
from source.graph import AgentRunner


_service = None
_agent = None
_current_persona = None


async def _ensure_service():
    global _service
    if _service is None:
        _service = await GraphitiService.create(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    return _service


async def _create_agent_with_persona(persona_uuid: str):
    global _agent, _current_persona
    service = await _ensure_service()
    _current_persona = await service.get_persona_by_uuid(persona_uuid)
    _agent = AgentRunner(service, model="gpt-4.1")
    return _agent


async def _start_session(user_name: str, system_prompt: str):
    service = await _ensure_service()
    user_uuid = await service.get_or_create_user_uuid(user_name)
    return user_uuid, uuid.uuid4().hex


async def _chat(history, user_input, user_name, system_prompt, user_uuid, thread_id):
    global _agent, _current_persona
    if _agent is None or _current_persona is None:
        # Use yield instead of return for async generators
        yield history + [{"role": "assistant", "content": "❌ Please create and select an agent persona first using the controls above."}]
        return  # Use return without a value to exit the generator
    
    # Get AI name from current persona
    ai_name = _current_persona.get("full_name", _current_persona.get("name", "AI friend"))
    
    state = {
        "messages": [{"role": "user", "content": user_input}],
        "user_name": user_name,
        "user_node_uuid": user_uuid,
        "system_prompt": system_prompt,
        "ai_persona": _current_persona,
    }

    # Create initial history entry with messages format
    history_with_user = history + [{"role": "user", "content": user_input}]
    yield history_with_user
    
    # Stream the response
    async for partial_response in _agent.astream_response(state, thread_id=thread_id, ai_name=ai_name):
        # Update the last message to include assistant response
        history_with_user = history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": partial_response}
        ]
        yield history_with_user


async def create_persona(name, surname, age, profession, hobbies, additional_info):
    """Create a new agent persona"""
    try:
        service = await _ensure_service()
        print(f"Debug: Creating persona - {name} {surname}, {age}, {profession}")
        uuid = await service.create_agent_persona(name, surname, age, profession, hobbies, additional_info)
        print(f"Debug: Created persona with UUID: {uuid}")
        if uuid:
            # Refresh the personas list
            personas_update = await load_personas()
            return (
                personas_update,
                "✅ Persona created successfully!",
                "", "", "", "", "", ""  # Clear form fields
            )
        else:
            print("Debug: UUID is None/False")
            return gr.update(), "❌ Failed to create persona", name, surname, str(age), profession, hobbies, additional_info
    except Exception as e:
        print(f"Debug: Exception in create_persona: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return gr.update(), f"❌ Error: {str(e)}", name, surname, str(age), profession, hobbies, additional_info


async def load_personas():
    """Load all existing personas for the dropdown"""
    try:
        service = await _ensure_service()
        personas = await service.get_all_personas()
        print(f"Debug: Retrieved {len(personas) if personas else 0} personas")
        
        if personas:
            choices = [f"{p['name']} {p['surname']} - {p['profession']}" for p in personas]
            return gr.update(choices=choices, value=None)  # Set to None instead of empty string
        else:
            return gr.update(choices=[], value=None)
    except Exception as e:
        print(f"Error loading personas: {e}")
        import traceback
        traceback.print_exc()
        return gr.update(choices=[], value=None)


async def select_persona(persona_choice):
    """Select a persona and initialize the agent"""
    try:
        if not persona_choice:
            return "Please select a persona", gr.update(interactive=False), gr.update(value="No agent selected")
        
        service = await _ensure_service()
        personas = await service.get_all_personas()
        
        # Find the selected persona
        selected_persona = None
        for persona in personas:
            persona_display = f"{persona['name']} {persona['surname']} - {persona['profession']}"
            if persona_display == persona_choice:
                selected_persona = persona
                break
        
        if selected_persona:
            await _create_agent_with_persona(selected_persona['uuid'])
            persona_info = f"""**Selected Agent:** {selected_persona['name']} {selected_persona['surname']}
**Age:** {selected_persona['age']} | **Profession:** {selected_persona['profession']}
**Hobbies:** {selected_persona['hobbies']}"""
            
            status = f"✅ Agent {selected_persona['name']} {selected_persona['surname']} is ready to chat!"
            
            return persona_info, gr.update(interactive=True), gr.update(value=status)
        else:
            return "❌ Persona not found", gr.update(interactive=False), gr.update(value="No agent selected")
    except Exception as e:
        error_msg = f"❌ Error selecting persona: {str(e)}"
        return error_msg, gr.update(interactive=False), gr.update(value="Error selecting agent")


def build_app():
    with gr.Blocks(title="Graphiti Memory Chat with Agent Personas") as demo:
        gr.Markdown("# Graphiti Memory Chat with Agent Personas")
        
        # Persona Management Section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Agent Management")
                with gr.Row():
                    persona_dropdown = gr.Dropdown(
                        label="Select Agent",
                        choices=[],
                        value=None,  # Set to None instead of empty string
                        interactive=True,
                        allow_custom_value=False
                    )
                    load_btn = gr.Button("🔄", variant="secondary", scale=0)
                
                select_btn = gr.Button("✅ Activate Agent", variant="primary")
                agent_status = gr.Textbox(label="Agent Status", value="No agent selected", interactive=False)
                persona_info = gr.Markdown("", visible=False)
                
            with gr.Column(scale=1):
                gr.Markdown("### Create New Agent")
                with gr.Row():
                    name_input = gr.Textbox(label="Name", placeholder="e.g., Kai", scale=1)
                    surname_input = gr.Textbox(label="Surname", placeholder="e.g., Smith", scale=1)
                    age_input = gr.Number(label="Age", value=25, minimum=1, maximum=150, scale=0)
                
                profession_input = gr.Textbox(label="Profession", placeholder="e.g., Software Engineer")
                hobbies_input = gr.Textbox(label="Hobbies", placeholder="e.g., reading, coding, hiking")
                additional_input = gr.Textbox(label="Additional Info", placeholder="Any other relevant information...")
                
                with gr.Row():
                    create_btn = gr.Button("🎭 Create Agent", variant="primary")
                    create_status = gr.Markdown("")

        gr.Markdown("---")
        
        # Chat Interface Section
        gr.Markdown("## Chat Interface")
        
        with gr.Row():
            user_name = gr.Textbox(label="Your name", value="Sadettin", scale=1)
            system_prompt = gr.Textbox(
                label="System prompt (editable)",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=3,
                scale=2,
            )

        chatbot = gr.Chatbot(height=420, type='messages')
        msg = gr.Textbox(label="Message", placeholder="Type your message here...", interactive=False)
        with gr.Row():
            send = gr.Button("Send", variant="primary", interactive=False)
            clear = gr.Button("New Session", variant="secondary")

        user_uuid = gr.State("")
        thread_id = gr.State("")

        # Event handlers
        load_btn.click(load_personas, outputs=[persona_dropdown])
        
        create_btn.click(
            create_persona,
            inputs=[name_input, surname_input, age_input, profession_input, hobbies_input, additional_input],
            outputs=[persona_dropdown, create_status, name_input, surname_input, age_input, profession_input, hobbies_input, additional_input]
        )
        
        select_btn.click(
            select_persona,
            inputs=[persona_dropdown],
            outputs=[persona_info, msg, agent_status]
        )

        # Update send button interactivity when message input changes
        msg.change(
            lambda msg_text, status: gr.update(interactive=bool(msg_text.strip() and "ready to chat" in status.lower())),
            inputs=[msg, agent_status],
            outputs=[send]
        )

        async def on_mount():
            uid, tid = await _start_session(user_name.value, system_prompt.value)
            personas_update = await load_personas()
            return {user_uuid: uid, thread_id: tid, persona_dropdown: personas_update}

        demo.load(on_mount, inputs=None, outputs=[user_uuid, thread_id, persona_dropdown])

        async def on_send(message, name, prompt, uid, tid, history):
            if not uid:
                uid, tid = await _start_session(name, prompt)
            
            # Clear the input message immediately
            yield history, "", uid, tid
            
            # Stream the response
            async for updated_history in _chat(history, message, name, prompt, uid, tid):
                yield updated_history, "", uid, tid

        send.click(
            on_send,
            inputs=[msg, user_name, system_prompt, user_uuid, thread_id, chatbot],
            outputs=[chatbot, msg, user_uuid, thread_id],
        )

        async def on_clear(name, prompt):
            uid, tid = await _start_session(name, prompt)
            return [], uid, tid

        clear.click(on_clear, inputs=[user_name, system_prompt], outputs=[chatbot, user_uuid, thread_id])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch()