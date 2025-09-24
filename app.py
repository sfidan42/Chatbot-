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


async def _ensure_service_and_agent():
    global _service, _agent
    if _service is None:
        _service = await GraphitiService.create(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        _agent = AgentRunner(_service, model="gemma3:4b")
    return _service, _agent


async def _start_session(user_name: str, system_prompt: str):
    service, _ = await _ensure_service_and_agent()
    user_uuid = await service.get_or_create_user_uuid(user_name)
    return user_uuid, uuid.uuid4().hex


async def _chat(history, user_input, user_name, system_prompt, user_uuid, thread_id):
    _, agent = await _ensure_service_and_agent()
    state = {
        "messages": [{"role": "user", "content": user_input}],
        "user_name": user_name,
        "user_node_uuid": user_uuid,
        "system_prompt": system_prompt,
    }
    
    # Create initial history entry with messages format
    history_with_user = history + [{"role": "user", "content": user_input}]
    yield history_with_user
    
    # Stream the response
    async for partial_response in agent.astream_response(state, thread_id=thread_id):
        # Update the last message to include assistant response
        history_with_user = history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": partial_response}
        ]
        yield history_with_user


def build_app():
    with gr.Blocks(title="Graphiti Memory Chat") as demo:
        gr.Markdown("## Graphiti Memory Chat")

        with gr.Row():
            user_name = gr.Textbox(label="Your name", value="Sadettin", scale=1)
            system_prompt = gr.Textbox(
                label="System prompt (editable)",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=6,
                scale=2,
            )

        chatbot = gr.Chatbot(height=420, type='messages')
        msg = gr.Textbox(label="Message")
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("New Session", variant="secondary")

        user_uuid = gr.State("")
        thread_id = gr.State("")

        async def on_mount():
            uid, tid = await _start_session(user_name.value, system_prompt.value)
            return {user_uuid: uid, thread_id: tid}

        demo.load(on_mount, inputs=None, outputs=[user_uuid, thread_id])

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
