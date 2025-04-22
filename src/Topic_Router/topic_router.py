from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
# Load BERTopic model once

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # or HF path

topic_model = BERTopic.load("Jsevere/bertopic-admissions-mmr-keybert", embedding_model = embedding_model)  # or HF path

# Prewritten response map
topic_to_response = {
    0: "For scholarship questions, please visit the Financial Aid Office or check the scholarship portal.",
    1: "You can request official transcripts through the Registrars portal.",
    2: "International students should visit the International Services page for visa and document guidance.",
    3: "For help with admissions, contact the University Admissions Office here: 978-542-6200.",
    4: "For additional info about the application process (for your specific program), please check the Admissions page.",
}

def classify_topic_and_get_response(user_input: str) -> str:
    topic_id, _ = topic_model.transform([user_input])
    topic_id = topic_id[0]

    return topic_to_response.get(
        topic_id,
        "We’re not sure how to help with that — please contact the Admissions Office for personalized assistance."
    )
