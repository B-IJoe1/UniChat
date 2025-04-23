import chainlit as cl
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain_community.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_hub import HuggingFacePipeline

# (your existing pipeline functions here)
# — load_llm()
# — custom_prompt()
# — qa_bot()
# — custom_chain()

@cl.on_chat_start
async def start():
    # 1) build your QA components once
    retriever, memory, llm, prompt = qa_bot()

    # 2) define an async wrapper that Chainlit will call per‐message
    async def run_chain(user_input, callbacks=None):
        # call your custom_chain, but await its .acall to get streaming support
        # note: custom_chain builds a CRChain; we call .acall on it
        cr_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            prompt=prompt,
            return_source_documents=True,
        )
        return await cr_chain.acall(
            {"question": user_input, "context": ""},  # context is injected inside custom_chain
            callbacks=callbacks
        )

    # 3) stash it in the user session
    cl.user_session.set("chain", run_chain)

    # 4) send your welcome
    msg = cl.Message(content="Starting your gen AI bot!...")
    await msg.send()
    msg.content = "Welcome to Demo Bot! Ask your question here:"
    await msg.update()


@cl.on_message
async def main(message):
    # 1) pull back the wrapper
    chain = cl.user_session.get("chain")
    if not chain:
        await cl.Message(content="Bot not initialized.").send()
        return

    # 2) prepare a LangChain callback for streaming & prefix
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    # 3) run the chain, passing the callback
    res = await chain(message.content, callbacks=[cb])

    # 4) format & send
    text = res["result"]
    docs = res.get("source_documents", []) or res.get("documents", [])
    if docs:
        text += "\n\nSources:\n" + "\n".join(d.page_content for d in docs)
    else:
        text += "\n\nNo sources found."

    await cl.Message(content=text).send()