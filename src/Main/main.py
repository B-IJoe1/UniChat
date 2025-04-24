import chainlit as cl
from Llm_pipeline.pipeline import create_qa_chain
from Llm_pipeline.pipeline import qa_bot_answer


@cl.on_chat_start
async def start():

    qa_chain = create_qa_chain()
    welcome_message = cl.Message(content = "Starting the bot....")
    await welcome_message.send()

    msg = cl.Message(content="Welcome! Ask me anything:")
    await msg.update()

    cl.user_session.set("chain", qa_chain)


@cl.on_message
async def main(message):
    qa_chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    #Response geenration 
    response = await qa_chain.acall(message.content, callbacks=[cb])
    bot_answer = response["result"]
    source_documents = response.get("source_documents", [])
    if source_documents:
        answer += "\n\nSources:\n" + "\n".join(d.page_content for d in source_documents)
    await cl.Message(content=bot_answer).send()