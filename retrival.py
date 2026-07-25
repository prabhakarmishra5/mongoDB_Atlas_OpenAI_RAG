import argparse
from typing import Optional, Sequence

from openai import OpenAIError
from pymongo.errors import (
    OperationFailure,
    PyMongoError,
    ServerSelectionTimeoutError,
)

from rag_common import (
    CHAT_MODEL,
    COLLECTION_NAME,
    DATABASE_NAME,
    VECTOR_INDEX_NAME,
    VECTOR_PATH,
    configure_logger,
    confirm_yes,
    connect_to_collection,
    create_openai_client,
    embed_text,
    env_int,
    prompt_non_empty,
)

logger = configure_logger(__name__)


class RetrievalError(RuntimeError):
    """Raised when the retrieval pipeline cannot produce an answer."""


openai_client = None
mongo_client = None
collection = None


def get_openai_client():
    """Create and cache the OpenAI client on demand."""
    global openai_client
    if openai_client is None:
        openai_client = create_openai_client()
    return openai_client


def get_mongo_collection():
    """Create and cache the MongoDB collection on demand."""
    global mongo_client, collection
    if collection is None:
        mongo_client, collection = connect_to_collection(
            DATABASE_NAME,
            COLLECTION_NAME,
        )
    return collection


def close_clients():
    """Close any open MongoDB client connection."""
    global mongo_client, collection
    if mongo_client is not None:
        try:
            mongo_client.close()
        except Exception as exc:
            # Never let cleanup mask the original failure.
            logger.warning(f"Failed to close MongoDB connection: {str(exc)}")
        finally:
            mongo_client = None
            collection = None


def prompt_for_question():
    """Prompt the user for a question to send to the RAG system."""
    return prompt_non_empty(
        "Enter your question: ",
        "Please enter a non-empty question.",
    )


def confirm_proceed():
    """Ask the user to confirm they want to proceed with the current question."""
    return confirm_yes("Type Yes to continue: ")


def get_brain_response(user_query):
    """
    Retrieve context from MongoDB vector search and generate LLM response.

    Args:
        user_query (str): The user's question

    Returns:
        str: The LLM-generated response

    Raises:
        ValueError: If user_query is empty or a search setting is malformed
        ConnectionError: If MongoDB Atlas is unreachable
        RetrievalError: If the MongoDB query or an OpenAI call fails
    """
    if not user_query or not user_query.strip():
        raise ValueError("User query cannot be empty")

    user_query = user_query.strip()
    num_candidates = env_int("NUM_CANDIDATES", 10)
    result_limit = env_int("RESULT_LIMIT", 2)

    try:
        openai_client_instance = get_openai_client()
        collection_instance = get_mongo_collection()

        query_embedding = embed_text(openai_client_instance, user_query)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": VECTOR_PATH,
                    "queryVector": query_embedding,
                    "numCandidates": num_candidates,
                    "limit": result_limit,
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

        if not completion.choices:
            raise RetrievalError(f"{CHAT_MODEL} returned no completion choices")

        answer = completion.choices[0].message.content
        if answer is None or not answer.strip():
            raise RetrievalError(f"{CHAT_MODEL} returned an empty answer")

        return answer

    except ServerSelectionTimeoutError as exc:
        raise ConnectionError(f"Failed to connect to MongoDB Atlas: {exc}") from exc
    except OperationFailure as exc:
        raise RetrievalError(f"MongoDB query failed: {exc}") from exc
    except PyMongoError as exc:
        raise RetrievalError(f"MongoDB request failed: {exc}") from exc
    except OpenAIError as exc:
        raise RetrievalError(f"OpenAI request failed: {exc}") from exc


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
    except KeyboardInterrupt:
        print("Operation cancelled by user.")
        return 130
    except (RetrievalError, ConnectionError, ValueError) as exc:
        logger.error(str(exc))
        return 1
    except Exception as exc:
        logger.exception(f"Unexpected failure: {str(exc)}")
        return 1
    finally:
        close_clients()


if __name__ == "__main__":
    raise SystemExit(main())
