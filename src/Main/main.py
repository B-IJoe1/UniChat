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


@cl.on_chat_start
async def start():
    try:
        logger.info("Initializing the QA bot...")
        # Initialize components from qa_bot
        retriever, memory, llm, prompt = qa_bot()

        # Define the chain function for processing messages
        async def run_custom_chain(user_input):
            return custom_chain(user_input, retriever, memory, llm, prompt)

        # Set it in the session BEFORE anything else can throw
        cl.user_session.set("chain", run_custom_chain)
        logger.info("Chain successfully set in the user session.")

        # Send welcome message
        msg = cl.Message(content="Starting your gen AI bot!...")
        await msg.send()
        msg.content = "Welcome to Uni Chat! Ask your question here:"
        await msg.update()

        logger.info("QA bot initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing the QA bot: {e}")
        msg = cl.Message(content=f"Error initializing the bot: {e}. Please try again.")
        await msg.send()
        
        

@cl.on_message
async def main(message):
    try:
        logger.info(f"Received message: {message.content}")
        chain = cl.user_session.get("chain")

        if not chain:
            raise ValueError("Chain not initialized. Please start the bot first.")
        
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