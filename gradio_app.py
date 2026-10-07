"""
gradio_app.py
=============
Interactive Web UI for GenAI StudyLab using Gradio.
Allows testing Basic Prompting, RAG, Agent with tools, and Self-Consistency.

Run:
python gradio_app.py
"""

import gradio as gr
from src.agents import build_single_agent
from src.chains import basic_chain, build_rag_chain
from src.rag import get_retriever
from src.self_consistency import self_consistent_answer

# Initialize core components
retriever, _, chunks = get_retriever()
rag_chain = build_rag_chain(retriever)
agent = build_single_agent(retriever)


def process_query(question: str, mode: str):
    if not question.strip():
        return "Please enter a question or problem."

    try:
        if mode == "RAG":
            return rag_chain.invoke(question)
        elif mode == "Basic Prompt":
            return basic_chain.invoke({"question": question})
        elif mode == "Single Agent":
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})
            return result["messages"][-1].content
        elif mode == "Self-Consistency":
            res = self_consistent_answer(question)
            output = f"Consensus Winner: {res['answer']} ({res['votes']}/{len(res['candidates'])} votes)\n\n"
            output += "Candidate Samples:\n"
            for i, sample in enumerate(res["samples"], 1):
                output += f"\n--- Sample {i} ---\n{sample.strip()}\n"
            return output
        else:
            return f"Unknown mode: {mode}"
    except Exception as exc:
        return f"Error: {exc}\n\nMake sure Ollama is running with required models."


def create_ui():
    with gr.Blocks(title="GenAI StudyLab") as demo:
        gr.Markdown(
            "# GenAI StudyLab - Interactive UI\n"
            "Select an execution mode and test how different GenAI techniques operate locally."
        )

        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Question / Problem",
                    placeholder="e.g. How many annual leave days do employees receive?",
                    lines=3,
                )
                mode_radio = gr.Radio(
                    choices=["RAG", "Basic Prompt", "Single Agent", "Self-Consistency"],
                    value="RAG",
                    label="Execution Mode",
                )
                submit_btn = gr.Button("Submit", variant="primary")

                gr.Examples(
                    examples=[
                        ["How many annual leave days do employees receive?", "RAG"],
                        ["Explain foundation models in simple terms.", "Basic Prompt"],
                        ["What is the training budget, and what is 18 * 7?", "Single Agent"],
                        [
                            "When I was 6, my sister was half my age. Now I am 70. How old is my sister?",
                            "Self-Consistency",
                        ],
                    ],
                    inputs=[question_input, mode_radio],
                )

            with gr.Column(scale=3):
                output_box = gr.Textbox(
                    label="Model Output",
                    lines=12,
                    interactive=False,
                )

        submit_btn.click(
            fn=process_query,
            inputs=[question_input, mode_radio],
            outputs=[output_box],
        )

    return demo


if __name__ == "__main__":
    ui = create_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860)
