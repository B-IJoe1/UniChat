import chainlit as cl
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Llm_pipeline.pipeline import create_qa_chain, qa_bot_answer, load_llm, custom_prompt
from Topic_Router.topic_router import topic_to_response


#Automatically start the chat when the app is launched
@cl.on_chat_start
async def start():

    qa_chain = create_qa_chain(load_llm=load_llm, custom_prompt=custom_prompt)
    print("LLM Loader:", load_llm)
    print("Prompt Template:", custom_prompt)
    cl.user_session.set("chain", qa_chain) 
    await cl.Message(content = "Welcome! Ask me anything:").send()


#Specific to each user session where multiple users interace w/the bot simultaneously


# This is the tool that will be called by the LLM 
#@cl.step(type="tool")
async def process_tool(message: cl.Message, qa_chain):
    #qa_chain = cl.user_session.get("chain")
    #cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    #cb.answer_reached = True

    #waiting to call the chain which includes the LLM and the retriever
    response = await qa_chain.ainvoke({"input": message.content})
    return response

#This will display the final answer from the bot
@cl.on_message
async def main(message: cl.Message):
    qa_chain = cl.user_session.get("chain")
    tool_response = await process_tool(message, qa_chain)

    bot_response = await qa_bot_answer(message.content, qa_chain, tool_response, topic_to_response)

    await cl.Message(content=bot_response).send()