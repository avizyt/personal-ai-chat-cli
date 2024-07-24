import typer
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import json
from typing import List, Dict

app = typer.Typer()
console = Console()

# Initialize smolLM model
model_name = "HuggingFaceTB/SmolLM-360M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

# Create a text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256,
    truncation=True,
    temperature=0.7,
    do_sample=True,
    repetition_penalty=1.2,
)

# Create a LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Create a prompt template
template = """
You are a helpful assistant. Provide a concise and informative answer to the following query:

Query: {query}

Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["query"])

# Create a LangChain
chain = prompt | llm


def generate_response(query: str) -> str:
    try:
        with console.status("Thinking...", spinner="dots"):
            response = chain.invoke(query)
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Sorry, I encountered an issue. Please try rephrasing your query."


def save_conversation(conversation: List[Dict[str, str]]):
    """Save the conversation history to a JSON file."""
    filename = typer.prompt(
        "Enter a filename to save the conversation (without extension)"
    )
    try:
        with open(f"{filename}.json", "w") as f:
            json.dump(conversation, f, indent=2)
        console.print(f"Conversation saved to {filename}.json", style="green")
    except Exception as e:
        print(f"An error occurred while saving: {e}")


@app.command()
def start():
    console.print(Panel.fit("Hi, I'm your Personal AI!", style="bold magenta"))
    conversation = []

    while True:
        console.print("How may I help you?", style="cyan")
        query = typer.prompt("You")

        if query.lower() in ["exit", "quit", "bye"]:
            break

        response = generate_response(query)
        conversation.append({"role": "user", "content": query})
        conversation.append({"role": "assistant", "content": response})

        console.print(Panel(Markdown(response), title="Assistant", expand=False))
        while True:
            console.print(
                "\nCHoose an action:",
                style="bold yellow",
            )
            console.print(
                "1. follow-up\n2. new-query\n3. end-chat\n4. save-and-exit",
                style="yellow",
            )
            action = typer.prompt("Enter the nuber corresponding to your choice.")

            if action == "1":
                follow_up = typer.prompt("Follow-up question")
                query = follow_up.lower()
                response = generate_response(query)

                conversation.append({"role": "user", "content": query})
                conversation.append({"role": "assistant", "content": response})

                console.print(
                    Panel(Markdown(response), title="Assistant", expand=False)
                )
            elif action == "2":
                new_query = typer.prompt("New query")
                query = new_query.lower()
                response = generate_response(query)

                conversation.append({"role": "user", "content": query})
                conversation.append({"role": "assistant", "content": response})

                console.print(
                    Panel(Markdown(response), title="Assistant", expand=False)
                )
            elif action == "3":
                return
            elif action == "4":
                save_conversation(conversation)
                return
            else:
                console.print(
                    "Invalid choice. Please choose a valid option.", style="red"
                )

    if typer.confirm("Would you like to save this conversation?"):
        save_conversation(conversation)

    console.print("Good Bye!! Happy Hacking", style="bold green")


if __name__ == "__main__":
    app()
