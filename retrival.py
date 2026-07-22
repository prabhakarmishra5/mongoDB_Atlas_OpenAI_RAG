import argparse
import logging
import os
import urllib.parse
from typing import Optional, Sequence

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
DATABASE_NAME = os.environ.get("DATABASE_NAME", "rag_database")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "knowledge_base")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
VECTOR_INDEX_NAME = os.environ.get("VECTOR_INDEX_NAME", "vector_index")
VECTOR_PATH = os.environ.get("VECTOR_PATH", "text_embedding")
NUM_CANDIDATES = int(os.environ.get("NUM_CANDIDATES", "10"))
RESULT_LIMIT = int(os.environ.get("RESULT_LIMIT", "2"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

openai_client = None
mongo_client = None
collection = None


def validate_credentials():
    """Validate that all required environment variables are set."""
    required_vars = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "MONGODB_USERNAME": MONGODB_USERNAME,
        "MONGODB_PASSWORD": MONGODB_PASSWORD,
    }
    missing = [name for name, value in required_vars.items() if not value]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    logger.info("✓ All credentials validated")


def build_mongodb_uri():
    """Build MongoDB connection string with properly encoded credentials."""
    username = urllib.parse.quote_plus(MONGODB_USERNAME)
    password = urllib.parse.quote_plus(MONGODB_PASSWORD)
    return f"mongodb+srv://{username}:{password}@{MONGODB_CLUSTER}/?appName=Cluster0&compressors=zlib"


def get_openai_client():
    """Create and cache the OpenAI client on demand."""
    global openai_client
    if openai_client is None:
        validate_credentials()
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return openai_client


def get_mongo_collection():
    """Create and cache the MongoDB collection on demand."""
    global mongo_client, collection
    if collection is None:
        validate_credentials()
        mongo_uri = build_mongodb_uri()
        mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command("ping")
        db = mongo_client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        logger.info("✓ Connected to MongoDB Atlas")
    return collection


def close_clients():
    """Close any open MongoDB client connection."""
    global mongo_client, collection
    if mongo_client is not None:
        mongo_client.close()
        mongo_client = None
        collection = None


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
    if not user_query or not user_query.strip():
        raise ValueError("User query cannot be empty")

    user_query = user_query.strip()

    try:
        openai_client_instance = get_openai_client()
        collection_instance = get_mongo_collection()

        embeddings_response = openai_client_instance.embeddings.create(
            input=user_query,
            model=EMBEDDING_MODEL,
        )
        query_embedding = embeddings_response.data[0].embedding

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

        results = list(collection_instance.aggregate(pipeline))

        if not results:
            return "No relevant documents found in the knowledge base."

        context = "\n".join([doc.get("text", "") for doc in results])

        if not context.strip():
            return "Retrieved documents contain no text content."

        system_prompt = f"Answer the query using only this context:\n{context}"

        completion = openai_client_instance.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
        )

        return completion.choices[0].message.content

    except ServerSelectionTimeoutError as exc:
        raise ConnectionError("Failed to connect to MongoDB Atlas") from exc
    except OperationFailure as exc:
        raise Exception(f"MongoDB query failed: {str(exc)}")
    except Exception as exc:
        raise Exception(f"Error in get_brain_response: {str(exc)}")


def build_parser() -> argparse.ArgumentParser:
    """Create a CLI parser for non-interactive execution in CI/CD."""
    parser = argparse.ArgumentParser(description="Query the MongoDB RAG system")
    parser.add_argument("--question", help="Question to answer")
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip the confirmation prompt",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the retrieval flow from the command line."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        user_question = args.question or prompt_for_question()
        if not args.no_confirm and not confirm_proceed():
            print("Operation cancelled by user.")
            return 0

        response = get_brain_response(user_question)
        print(f"Query: {user_question}\n")
        print(f"Response: {response}")
        return 0
    except Exception as exc:
        print(f"Error: {str(exc)}")
        return 1
    finally:
        close_clients()


if __name__ == "__main__":
    raise SystemExit(main())
