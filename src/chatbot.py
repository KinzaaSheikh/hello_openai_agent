import os
from dotenv import load_dotenv
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from openai.types.responses import ResponseTextDeltaEvent
from agents.run import RunConfig

# Load environment variables from .env file
load_dotenv()

# Create an instance of the OpenAI API
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is set
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set")

@cl.on_chat_start
async def start():
    # Step 1: Set provider
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )

    # Step 2: Set model
    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    )

    # Step 3: Set config (defined at run level)
    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    # Step 4: Set agent
    book_agent: Agent = Agent(
        name="Book Recommender Agent",
        instructions="You are a helpful agent that suggests books to read based on the user's prompt.",
        model=model
        )
    
    cl.user_session.set("book_agent", book_agent)
    cl.user_session.set("config", config)
    cl.user_session.set("history", [])

    await cl.Message(
        content="Welcome to the Hyper-specific book recommendation chatbot! How can I assist you today?"
        ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    book_agent: Agent = cl.user_session.get("book_agent")
    config: RunConfig = cl.user_session.get("config")
    history = cl.user_session.get("history")

    msg = cl.Message(content="")
    await msg.send()
       
   # Standard Interface [{role: user, content: message.content}]
    history.append({"role": "user", "content": message.content})

    result = Runner.run_streamed(
        book_agent,
        input=history,
        run_config=config,
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)