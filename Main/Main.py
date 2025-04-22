from Llm_pipeline.pipeline import qa_bot, custom_chain
import chainlit as cl
import logging
import toml

#Configure logging to track errors and debug issues
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

config_path = "/Users/josephsevere/Downloads/CampusQuest/.chainlit/config.toml"
try:
    config = toml.load(config_path)
    logger.debug(f"Loaded config: {config}")
except Exception as e:
    logger.error(f"Error loading config.toml: {e}")

 #chainlit code
@cl.on_chat_start
async def start():
    try:
        retriever, memory, llm, prompt = qa_bot()
        
        async def run_custom_chain(user_input):
            return custom_chain(user_input, retriever, memory, llm, prompt)
        #logger.info("Starting the chat...")
        cl.user_session.set("chain", run_custom_chain)

        msg = cl.Message(content="Starting your gen AI bot!...")
        await msg.send()
        msg.content = "Welcome to Uni Chat!. Ask your question here:"
        await msg.update()
        logger.info("QA bot initialized succefully.")
    except Exception as e:
        logger.error(f"Error initializing the QA bot: {e}")
        msg = cl.Message(content=f"Error initializing the bot{e}. Please try again.")
        await msg.send()
        



    # Set the chain in the user session
    # This allows us to access the chain in the on_message function
    # so we can call it with the user's message and get the answer.
    cl.user_session.set("chain", run_custom_chain)
@cl.on_message
async def main(message):
    try:
        logger.info(f"Received message: {message.content}")
        chain = cl.user_session.get("chain")
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True

        res = await chain.acall(message.content, callbacks=[cb])
        answer = res["result"]
        sources = res.get("documents", [])

        if sources:
            answer += "\nSources:\n" + "\n".join([str(source) for source in sources])
        else:
            answer+= "\nNo sources found or topics. Please contact the Admissions Office for personalized assistance."
            #fallback = classify_topic_and_get_response(message.content)
            #answer = fallback

        await cl.Message(content=answer).send()
        logger.info("Response sent to the user.")
    except Exception as e:
        # Handle any errors that occur during processing
        logger.error(f"Error processing the message: {e}")
        await cl.Message(content=f"Error processing your message: {e}").send()