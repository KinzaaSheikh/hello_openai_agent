import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
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
    agent1: Agent = Agent(
        name="Book Recommender Agent",
        instructions="You are a helpful agent that suggests books to read based on the user's prompt.",
        model=model
        )
    
    @cl.on_chat_start
    async def handle_chat_start():
        cl.user_session.set("history", [])
        await cl.Message(
            content="Welcome to the Hyper-specific book recommendation chatbot! How can I assist you today?"
            ).send()
    
    @cl.on_message
    async def handle_message(message: cl.Message):

        history = cl.user_session.get("history") or []
        # Standard Interface [{role: user, content: message.content}]
        history.append({"role": "user", "content": message.content})
        result = await Runner.run(
            agent1,
            input=history,
            run_config=config,
        )

        history.append({"role": "assistant", "content": result.final_output})
        cl.user_session.set("history", history)

        await cl.Message(content=result.final_output).send()

    """"Set up the chat session when a user connects"""
    # # Initial an empty chat history in the session
    # cl.user_session.set("chat_history", [])

    # # Set the initial prompt for the chat session
    # cl.user_session.set("config", config)
    # cl.user_session.set("agent", agent)

    # # Set the initial prompt for the chat session
    # await cl.Message(
    #     content="Welcome to the Hyper-specific book recommendation chatbot! How can I assist you today?"
    #     ).send()

    # @cl.on_message
    # async def main(message: cl.Message):
    #     """Process incoming messages and generate responses."""
    #     # Send a thinking message
    #     msg = cl.Message(content="Thinking...")
    #     await msg.send()

    #     agent: Agent = cast(Agent, cl.user_session.get("agent"))
    #     config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    #     # Retrieve the chat history from the session.
    #     history = cl.user_session.get("chat_history") or []
        
    #     # Append the user's message to the history.
    #     history.append({"role": "user", "content": message.content})
        

    #     try:
    #         print("\n[CALLING_AGENT_WITH_CONTEXT]\n", history, "\n")
    #         result = Runner.run_sync(starting_agent = agent,
    #                     input=history,
    #                     run_config=config)
            
    #         response_content = result.final_output
            
    #         # Update the thinking message with the actual response
    #         msg.content = response_content
    #         await msg.update()
        
    #         # Update the session with the new history.
    #         cl.user_session.set("chat_history", result.to_input_list())
            
    #         # Optional: Log the interaction
    #         print(f"User: {message.content}")
    #         print(f"Assistant: {response_content}")
            
    #     except Exception as e:
    #         msg.content = f"Error: {str(e)}"
    #         await msg.update()
    #         print(f"Error: {str(e)}")