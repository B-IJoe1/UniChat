from typing import cast
import chainlit as cl
from langchain.schema import Runnable
from langchain.schema.runnable.config import RunnableConfig

from Llm_pipeline.pipeline import create_qa_chain, load_llm, custom_prompt
#from Topic_Router.topic_router import topic_to_response

@cl.on_chat_start
async def on_chat_start():
    # Set up your QA chain as a Runnable
    qa_chain = create_qa_chain(load_llm=load_llm, custom_prompt=custom_prompt)
    cl.user_session.set("runnable", qa_chain)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

    msg = cl.Message(content="")

    # Stream the answer as tokens/chunks
    async for chunk in runnable.astream(
        {"input": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
