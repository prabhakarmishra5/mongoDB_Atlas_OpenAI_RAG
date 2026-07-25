from datetime import datetime
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo.errors import (
    DuplicateKeyError,
    PyMongoError,
    ServerSelectionTimeoutError,
)

from rag_common import (
    COLLECTION_NAME,
    DATABASE_NAME,
    configure_logger,
    confirm_yes,
    connect_to_collection,
    create_openai_client,
    embed_text,
    read_json_dict,
    validate_credentials,
    write_json,
)

CHUNK_SIZE = 150
CHUNK_OVERLAP = 20

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR / "sourceFile"
LOG_FILE = SCRIPT_DIR / "ingestion_log.json"

logger = configure_logger(__name__)


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
    return read_json_dict(
        log_path or LOG_FILE,
        "Could not parse ingestion log, starting fresh",
    )


def save_ingestion_log(log_path=None, file_name=None, timestamp=None):
    """Persist the selected file name and ingestion timestamp."""
    path = Path(log_path or LOG_FILE)
    history = load_ingestion_log(path)
    history[file_name] = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    write_json(path, history)
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

    return confirm_yes(prompt)


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
        client, collection = connect_to_collection(DATABASE_NAME, COLLECTION_NAME)

        # Create index for vector search (if not exists)
        try:
            collection.create_index([("text_embedding", "2dsphere")])
            logger.info("Vector search index ready")
        except PyMongoError as exc:
            logger.warning(f"Index creation skipped: {str(exc)}")

        openai_client = create_openai_client()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_text(raw_text)
        logger.info(f"Text split into {len(chunks)} chunks")

        if not chunks:
            raise ValueError("Text produced no chunks to ingest")

        inserted_count = 0
        duplicate_count = 0
        failures = []
        for i, chunk in enumerate(chunks):
            try:
                embedding = embed_text(openai_client, chunk)

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
                duplicate_count += 1
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

        if inserted_count == 0:
            logger.warning(
                "No new chunks ingested: all %d chunks already existed in Atlas",
                duplicate_count,
            )
        else:
            logger.info(
                f"Successfully ingested {inserted_count}/{len(chunks)} "
                f"chunks into Atlas ({duplicate_count} already existed)!"
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
