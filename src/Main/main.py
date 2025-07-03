from typing import cast
import chainlit as cl
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables import Runnable
from Llm_pipeline.pipeline import create_qa_chain, load_llm, custom_prompt
#from Topic_Router.topic_router import topic_to_response

@cl.on_chat_start
async def on_chat_start(): 
    # Set up your QA chain as a Runnable
    qa_chain = create_qa_chain(load_llm=load_llm, custom_prompt=custom_prompt)
    cl.user_session.set("runnable", qa_chain) #This is almsost like an image of the chain that can be used later

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

    msg = cl.Message(content="")


    answer_prefix_tokens = ["FINAL", "ANSWER"]

   # Stream the answer as tokens/chunks
    async for chunk in runnable.astream(
        {"input": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler(stream_final_answer=True,
                                                                     answer_prefix_tokens=answer_prefix_tokens)]),
    ):
        # Check if the chunk is a string
        if isinstance(chunk, str):
            await msg.stream_token(chunk)
        elif isinstance(chunk, dict) and "text" in chunk:
            await msg.stream_token(chunk["text"])
        else:
            # Handle other cases or raise an error
            print(f"Unexpected chunk type: {type(chunk)}, value: {chunk}")
            await msg.stream_token(str(chunk))  # fallback to string conversion

    await msg.send()