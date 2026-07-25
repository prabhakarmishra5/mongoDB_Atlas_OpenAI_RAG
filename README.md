# MongoDB RAG

A retrieval-augmented generation (RAG) demo using MongoDB Atlas Vector Search, OpenAI embeddings, and a PDF ingestion pipeline.

## Project overview

This repository provides:

- `ingestion.py`: ingest PDF documents from `sourceFile/` into MongoDB Atlas using OpenAI embeddings
- `retrival.py`: query MongoDB and generate answers using OpenAI chat completions
- `evaluate_rag.py`: run sample questions and optionally save evaluation results to `evaluation_results.json`
- `tests/`: unit tests for ingestion and evaluation flows

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Environment variables

Create a `.env` file in the project root with the following values:

```env
OPENAI_API_KEY=your_openai_api_key
MONGODB_USERNAME=your_mongodb_username
MONGODB_PASSWORD=your_mongodb_password
MONGODB_CLUSTER=your_cluster_uri
```

All four values are required; there is no default cluster host. Never commit a real `.env` file — it is ignored by `.gitignore`.

## Ingestion

1. Place one or more PDF files into `sourceFile/`.
2. Run:

```bash
python ingestion.py
```

3. Select a PDF and confirm ingestion. The script splits text into chunks, generates embeddings, and stores them in MongoDB Atlas.

## Querying

Run the RAG retrieval flow:

```bash
python retrival.py
```

Enter a question and confirm to retrieve grounded results from the indexed documents.

## Evaluation

Run sample question evaluation and save results:

```bash
python evaluate_rag.py
```

The results are written to `evaluation_results.json` by default.

## Testing

Run unit tests with:

```bash
python -m unittest discover -s tests -v
```

## CI/CD

This repository includes a GitHub Actions workflow in [.github/workflows/ci.yml](.github/workflows/ci.yml) that installs dependencies and runs the test suite on every push and pull request.

## Non-interactive usage

You can run the retrieval flow in CI or automation without prompts:

```bash
python retrival.py --question "What does Atlas Vector Search do?" --no-confirm
```

## Notes

- `ingestion.py` uses `langchain_text_splitters.RecursiveCharacterTextSplitter` for chunking.
- `retrival.py` uses MongoDB Atlas vector search and OpenAI `gpt-4o-mini` for response generation.
- `evaluate_rag.py` can be extended to use a custom scoring callback.
