import json
import logging
import os
import urllib.parse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pymongo import MongoClient
from pymongo.errors import (
    DuplicateKeyError,
    PyMongoError,
    ServerSelectionTimeoutError,
)

# Configuration
load_dotenv()  # Load from .env file

MONGODB_USERNAME = os.environ.get("MONGODB_USERNAME")
MONGODB_PASSWORD = os.environ.get("MONGODB_PASSWORD")
MONGODB_CLUSTER = os.environ.get("MONGODB_CLUSTER", "cluster0.aefs3mv.mongodb.net")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

DATABASE_NAME = "rag_database"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 150
CHUNK_OVERLAP = 20

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR / "sourceFile"
LOG_FILE = SCRIPT_DIR / "ingestion_log.json"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_credentials():
    """Validate that all required environment variables are set."""
    required_vars = {
        "MONGODB_USERNAME": MONGODB_USERNAME,
        "MONGODB_PASSWORD": MONGODB_PASSWORD,
        "OPENAI_API_KEY": OPENAI_API_KEY,
    }

    missing = [var for var, val in required_vars.items() if not val]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

    logger.info("All credentials validated")


def build_connection_string():
    """Build MongoDB connection string with properly encoded credentials."""
    username = urllib.parse.quote_plus(MONGODB_USERNAME)
    password = urllib.parse.quote_plus(MONGODB_PASSWORD)
    return f"mongodb+srv://{username}:{password}@{MONGODB_CLUSTER}/?appName=Cluster0&compressors=zlib"


def get_pdf_files(source_dir):
    """Return only PDF files from the source directory."""
    source_path = Path(source_dir)
    if not source_path.exists() or not source_path.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_path}")

    return sorted(
        path
        for path in source_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def load_ingestion_log(log_path=None):
    """Load the ingestion history from disk if it exists."""
    path = Path(log_path or LOG_FILE)
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Could not parse ingestion log %s (%s), starting fresh", path, exc
        )
        return {}
    except OSError as exc:
        logger.warning(
            "Could not read ingestion log %s (%s), starting fresh", path, exc
        )
        return {}

    if not isinstance(data, dict):
        logger.warning(
            "Ingestion log %s contains %s instead of an object, starting fresh",
            path,
            type(data).__name__,
        )
        return {}

    return data


def save_ingestion_log(log_path=None, file_name=None, timestamp=None):
    """Persist the selected file name and ingestion timestamp."""
    path = Path(log_path or LOG_FILE)
    history = load_ingestion_log(path)
    history[file_name] = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
    except OSError as exc:
        raise OSError(f"Failed to write ingestion log to {path}: {exc}") from exc

    return history


def confirm_ingestion(file_name, last_ingested_at=None):
    """Ask the user to confirm whether to proceed with ingestion."""
    if last_ingested_at:
        prompt = (
            f"{file_name} was last ingested on {last_ingested_at}. "
            "Type Yes to re-ingest it, or anything else to skip: "
        )
    else:
        prompt = f"Proceed to ingest '{file_name}'? Type Yes to continue: "

    response = input(prompt).strip()
    return response.lower() == "yes"


def select_single_pdf(source_dir):
    """Prompt the user to choose exactly one PDF file from the source folder."""
    pdf_files = get_pdf_files(source_dir)
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {source_dir}")

    print("Available PDF documents:")
    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"{index}. {pdf_path.name}")

    while True:
        choice = input("Select one document by number: ").strip()
        try:
            selected_index = int(choice) - 1
        except ValueError:
            print("Please enter a valid number.")
            continue

        if 0 <= selected_index < len(pdf_files):
            return pdf_files[selected_index]

        print("Selection is out of range. Please choose one of the listed documents.")


def extract_text_from_pdf(pdf_path):
    """Read text content from a PDF file."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "Install the 'pypdf' package to ingest PDF documents."
        ) from exc

    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise ValueError(f"No readable text found in PDF: {pdf_path.name}")

    return text


def ingest_documents(raw_text):
    """
    Ingest documents into MongoDB Atlas with vector embeddings.

    Args:
        raw_text (str): Raw text to chunk and embed

    Returns:
        int: Number of chunks successfully ingested
    """
    validate_credentials()

    client = None
    try:
        # Connect to MongoDB Atlas
        mongo_uri = build_connection_string()
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")

        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        logger.info(f"Connected to MongoDB: {DATABASE_NAME}.{COLLECTION_NAME}")

        # Create index for vector search (if not exists)
        try:
            collection.create_index([("text_embedding", "2dsphere")])
            logger.info("Vector search index ready")
        except PyMongoError as exc:
            logger.warning(f"Index creation skipped: {str(exc)}")

        # Initialize OpenAI Client
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

        # Chunk text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_text(raw_text)
        logger.info(f"Text split into {len(chunks)} chunks")

        if not chunks:
            raise ValueError("Text produced no chunks to ingest")

        # Generate embeddings and ingest
        inserted_count = 0
        failures = []
        for i, chunk in enumerate(chunks):
            try:
                response = openai_client.embeddings.create(
                    input=chunk,
                    model=EMBEDDING_MODEL,
                )
                embedding = response.data[0].embedding

                result = collection.insert_one(
                    {
                        "chunk_id": i,
                        "text": chunk,
                        "text_embedding": embedding,
                    }
                )

                inserted_count += 1
                logger.debug(f"Inserted chunk {i}: {result.inserted_id}")

            except DuplicateKeyError:
                logger.warning(f"Chunk {i} already exists (skipped)")
            except Exception as exc:
                logger.exception(f"Failed to ingest chunk {i}: {str(exc)}")
                failures.append((i, exc))
                continue

        if failures and inserted_count == 0:
            first_index, first_exc = failures[0]
            raise RuntimeError(
                f"All {len(chunks)} chunks failed to ingest; "
                f"first failure on chunk {first_index}: {first_exc}"
            ) from first_exc

        if failures:
            logger.error(
                "%d/%d chunks failed to ingest (chunk ids: %s)",
                len(failures),
                len(chunks),
                ", ".join(str(index) for index, _ in failures),
            )

        logger.info(
            f"Successfully ingested {inserted_count}/{len(chunks)} chunks into Atlas!"
        )
        return inserted_count

    except ServerSelectionTimeoutError:
        logger.exception(
            "Failed to connect to MongoDB Atlas. Check your connection string."
        )
        raise
    except Exception as exc:
        logger.exception(f"Ingestion failed: {str(exc)}")
        raise
    finally:
        if client is not None:
            try:
                client.close()
                logger.info("MongoDB connection closed")
            except PyMongoError as exc:
                # Never let cleanup mask the original failure.
                logger.warning(f"Failed to close MongoDB connection: {str(exc)}")


def main():
    """Prompt for a single PDF, confirm the action, and ingest it."""
    selected_pdf = select_single_pdf(SOURCE_DIR)
    history = load_ingestion_log(LOG_FILE)
    last_ingested_at = history.get(selected_pdf.name)

    if not confirm_ingestion(selected_pdf.name, last_ingested_at):
        logger.info("Ingestion cancelled by user for %s", selected_pdf.name)
        return 0

    logger.info("Reading PDF content from %s", selected_pdf.name)
    raw_text = extract_text_from_pdf(selected_pdf)
    inserted_count = ingest_documents(raw_text)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        save_ingestion_log(LOG_FILE, selected_pdf.name, timestamp)
        logger.info("Logged ingestion for %s at %s", selected_pdf.name, timestamp)
    except OSError:
        # Ingestion already succeeded, so surface the bookkeeping failure
        # without discarding the result.
        logger.exception("Could not record ingestion history for %s", selected_pdf.name)
    print(f"Ingested {inserted_count} chunks from {selected_pdf.name}")
    return inserted_count


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user")
        raise SystemExit(130) from None
    except Exception as exc:
        logger.exception(f"Script failed: {str(exc)}")
        raise SystemExit(1) from exc
