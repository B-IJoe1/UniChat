import chainlit as cl
from Llm_pipeline.pipeline import qa_bot

# (your existing pipeline functions here)
# — load_llm()
# — custom_prompt()
# — qa_bot()
# — custom_chain()

@cl.on_chat_start
async def start():
    chain = qa_bot()
    cl.user_session.set("chain", chain)

    msg = cl.Message(content="Welcome! Ask me anything:")
    await msg.send()

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)
    res = await chain(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res.get("source_documents", [])
    if sources:
        answer += "\n\nSources:\n" + "\n".join(d.page_content for d in sources)
    await cl.Message(content=answer).send()