# app.py
import json
import gradio as gr
from pandasaitest import build_messages, conversation_history, generate_user_profile

def chat_with_karimi(user_input):
    if not user_input.strip():
        return "Please type a question 😊"
    reply = build_messages(user_input)
    if isinstance(reply, str) and reply.startswith("{\"error\""):
        # show system message for flagged input
        return "⚠️ Session flagged — user has detected that you are a robot."
    # Display conversation history
    chat_display = "\n\n".join(
        [f"👤 {c['user']}\n🤖 {c['assistant']}" for c in conversation_history]
    )
    return chat_display

def view_user_profile():
    profile = generate_user_profile(conversation_history)
    return json.dumps(profile, indent=2)

with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {max-width: 800px; margin: auto;}") as demo:
    gr.Markdown(
        """
        <div style='text-align: center'>
            <h1>🚗 Karimi — Your Vehicle Sales Assistant</h1>
            <p>Ask about vehicles, prices, and deals based on the latest inventory data.</p>
        </div>
        """
    )

    with gr.Tab("💬 Chat"):
        chatbot = gr.Textbox(label="Chat History", lines=15, interactive=False)
        user_input = gr.Textbox(label="Your Question to Karimi", placeholder="e.g., What’s the best deal on a Toyota Probox?")
        submit = gr.Button("Ask Karimi 🚀")
        submit.click(fn=chat_with_karimi, inputs=user_input, outputs=chatbot)
        submit.click(lambda: "", None, user_input)  # clears input

    with gr.Tab("🧠 User Profile"):
        profile_box = gr.Code(label="Generated User Profile (JSON)", language="json")
        profile_btn = gr.Button("Generate Profile 🪄")
        profile_btn.click(fn=view_user_profile, inputs=None, outputs=profile_box)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
