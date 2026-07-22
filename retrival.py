import logging
import os
import urllib.parse

from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MONGODB_USERNAME = os.environ.get("MONGODB_USERNAME")
MONGODB_PASSWORD = os.environ.get("MONGODB_PASSWORD")
MONGODB_CLUSTER = os.environ.get("MONGODB_CLUSTER", "cluster0.aefs3mv.mongodb.net")
DATABASE_NAME = "rag_database"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
VECTOR_INDEX_NAME = "vector_index"
VECTOR_PATH = "text_embedding"
NUM_CANDIDATES = 10
RESULT_LIMIT = 2

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Validate environment variables
def validate_credentials():
    required_vars = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "MONGODB_USERNAME": MONGODB_USERNAME,
        "MONGODB_PASSWORD": MONGODB_PASSWORD,
    }
    missing = [name for name, value in required_vars.items() if not value]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    logger.info("✓ All credentials validated")


# Create clients after validation
validate_credentials()
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# Build MongoDB connection string
def build_mongodb_uri():
    """Build MongoDB connection string with properly encoded credentials."""
    username = urllib.parse.quote_plus(MONGODB_USERNAME)
    password = urllib.parse.quote_plus(MONGODB_PASSWORD)
    return f"mongodb+srv://{username}:{password}@{MONGODB_CLUSTER}/?appName=Cluster0&compressors=zlib"


# Initialize MongoDB
try:
    mongo_uri = build_mongodb_uri()
    mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    mongo_client.admin.command("ping")
    db = mongo_client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    logger.info("✓ Connected to MongoDB Atlas")
except ServerSelectionTimeoutError:
    raise ConnectionError("Failed to connect to MongoDB Atlas")
except Exception as e:
    raise Exception(f"MongoDB initialization error: {str(e)}")


def prompt_for_question():
    """Prompt the user for a question to send to the RAG system."""
    while True:
        user_question = input("Enter your question: ").strip()
        if user_question:
            return user_question
        print("Please enter a non-empty question.")


def confirm_proceed():
    """Ask the user to confirm they want to proceed with the current question."""
    response = input("Type Yes to continue: ").strip()
    return response.lower() == "yes"


def get_brain_response(user_query):
    """
    Retrieve context from MongoDB vector search and generate LLM response.

    Args:
        user_query (str): The user's question

    Returns:
        str: The LLM-generated response

    Raises:
        ValueError: If user_query is empty or invalid
        Exception: If API calls fail
    """
    # Validate input
    if not user_query or not user_query.strip():
        raise ValueError("User query cannot be empty")

    user_query = user_query.strip()

    try:
        # 1. Vectorize user prompt
        embeddings_response = openai_client.embeddings.create(
            input=user_query, model=EMBEDDING_MODEL
        )
        query_embedding = embeddings_response.data[0].embedding

        # 2. Query MongoDB using $vectorSearch stage
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": VECTOR_PATH,
                    "queryVector": query_embedding,
                    "numCandidates": NUM_CANDIDATES,
                    "limit": RESULT_LIMIT,
                }
            },
            {"$project": {"_id": 0, "text": 1}},
        ]

        results = list(collection.aggregate(pipeline))

        # 3. Handle empty results
        if not results:
            return "No relevant documents found in the knowledge base."

        # 4. Combine matching documents into one context block
        context = "\n".join([doc.get("text", "") for doc in results])

        if not context.strip():
            return "Retrieved documents contain no text content."

        # 5. Generate grounded LLM response
        system_prompt = f"Answer the query using only this context:\n{context}"

        completion = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
        )

        return completion.choices[0].message.content

    except OperationFailure as e:
        raise Exception(f"MongoDB query failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Error in get_brain_response: {str(e)}")


# Execute RAG System
if __name__ == "__main__":
    try:
        user_question = prompt_for_question()
        if not confirm_proceed():
            print("Operation cancelled by user.")
            raise SystemExit(0)

        response = get_brain_response(user_question)
        print(f"Query: {user_question}\n")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Close MongoDB connection
        mongo_client.close()
