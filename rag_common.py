"""Shared configuration and helpers for the MongoDB Atlas RAG scripts."""

import json
import logging
import os
import urllib.parse
from pathlib import Path
from typing import Any, Iterable, List, Tuple, Union

from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient
from pymongo.collection import Collection

load_dotenv()

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
SERVER_SELECTION_TIMEOUT_MS = 5000
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def configure_logger(name: str) -> logging.Logger:
    """Apply the shared logging configuration and return a named logger."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    return logging.getLogger(name)


logger = configure_logger(__name__)


def env_int(name: str, default: int) -> int:
    """Read an integer setting, failing loudly on malformed values."""
    raw = os.environ.get(name)
    if raw is None:
        return default

    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def validate_credentials() -> None:
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


def build_mongodb_uri() -> str:
    """Build MongoDB connection string with properly encoded credentials."""
    username = urllib.parse.quote_plus(MONGODB_USERNAME)
    password = urllib.parse.quote_plus(MONGODB_PASSWORD)
    return (
        f"mongodb+srv://{username}:{password}@{MONGODB_CLUSTER}"
        "/?appName=Cluster0&compressors=zlib"
    )


def create_openai_client() -> OpenAI:
    """Validate credentials and return an OpenAI client."""
    validate_credentials()
    return OpenAI(api_key=OPENAI_API_KEY)


def connect_to_collection(
    database_name: str = DATABASE_NAME,
    collection_name: str = COLLECTION_NAME,
) -> Tuple[MongoClient, Collection]:
    """Connect to MongoDB Atlas and return the client and target collection."""
    validate_credentials()
    client = MongoClient(
        build_mongodb_uri(),
        serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
    )
    client.admin.command("ping")
    logger.info("✓ Connected to MongoDB Atlas: %s.%s", database_name, collection_name)
    return client, client[database_name][collection_name]


def embed_text(
    openai_client: OpenAI,
    text: str,
    model: str = EMBEDDING_MODEL,
) -> List[float]:
    """Return the embedding vector for a single piece of text."""
    response = openai_client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def prompt_non_empty(
    prompt: str,
    error_message: str = "Please enter a non-empty value.",
) -> str:
    """Prompt until the user provides a non-empty answer."""
    while True:
        answer = input(prompt).strip()
        if answer:
            return answer
        print(error_message)


def prompt_choice(prompt: str, valid_choices: Iterable[str], error_message: str) -> str:
    """Prompt until the user picks one of the allowed choices."""
    allowed = set(valid_choices)
    while True:
        answer = input(prompt).strip()
        if answer in allowed:
            return answer
        print(error_message)


def confirm_yes(prompt: str) -> bool:
    """Return True only when the user types 'yes' (case-insensitive)."""
    return input(prompt).strip().lower() == "yes"


def read_json_dict(path: Union[Path, str], default_message: str) -> dict:
    """Load a JSON object from disk, returning an empty dict when unusable."""
    json_path = Path(path)
    if not json_path.exists():
        return {}

    try:
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("%s (%s: %s)", default_message, json_path, exc)
        return {}
    except OSError as exc:
        logger.warning("Could not read %s (%s)", json_path, exc)
        return {}

    if not isinstance(data, dict):
        logger.warning(
            "%s (%s contains %s instead of an object)",
            default_message,
            json_path,
            type(data).__name__,
        )
        return {}

    return data


def write_json(path: Union[Path, str], payload: Any) -> Path:
    """Write JSON to disk, creating parent directories as needed."""
    json_path = Path(path)
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Failed to write JSON to {json_path}: {exc}") from exc

    return json_path
