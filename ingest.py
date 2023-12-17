"""Load html from files, clean up, split, ingest into Chroma."""
import logging
import os
import re
from parser import langchain_docs_extractor

from bs4 import BeautifulSoup, SoupStrainer
from langchain.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.indexes import SQLRecordManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from index import index
from chain import get_embeddings_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]
# Replace with your Chroma collection name
CHROMA_COLLECTION_NAME = os.environ["CHROMA_COLLECTION_NAME"]


def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
    title = soup.find("title")
    description = soup.find("meta", attrs={"name": "description"})
    html = soup.find("html")
    return {
        "source": meta["loc"],
        "title": title.get_text() if title else "",
        "description": description.get("content", "") if description else "",
        "language": html.get("lang", "") if html else "",
        **meta,
    }


def load_langchain_docs():
    return SitemapLoader(
        "https://python.langchain.com/sitemap.xml",
        filter_urls=["https://python.langchain.com/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def load_langsmith_docs():
    return RecursiveUrlLoader(
        url="https://docs.smith.langchain.com/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
    ).load()


def simple_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_api_docs():
    return RecursiveUrlLoader(
        url="https://api.python.langchain.com/en/latest/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://api.python.langchain.com/en/latest/_sources",
            "https://api.python.langchain.com/en/latest/_modules",
        ),
    ).load()


def preprocess_metadata(documents):
    for doc in documents:
        # Replace None values and ensure compatible types
        doc.metadata = {k: ("" if v is None else v)
                        for k, v in doc.metadata.items()}
        # Optionally filter out irrelevant fields
        # doc.metadata = {k: v for k, v in doc.metadata.items() if k in relevant_fields}


def ingest_docs():
    logger.info("Starting to load documents from documentation...")
    docs_from_documentation = load_langchain_docs()
    logger.info(
        f"Loaded {len(docs_from_documentation)} docs from documentation")

    logger.info("Starting to load documents from API...")
    docs_from_api = load_api_docs()
    logger.info(f"Loaded {len(docs_from_api)} docs from API")

    logger.info("Starting to load documents from Langsmith...")
    docs_from_langsmith = load_langsmith_docs()
    logger.info(f"Loaded {len(docs_from_langsmith)} docs from Langsmith")

    logger.info("Starting document text splitting...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=200)
    docs_transformed = text_splitter.split_documents(
        docs_from_documentation + docs_from_api + docs_from_langsmith
    )
    logger.info(
        f"Completed document text splitting. Number of documents after splitting: {len(docs_transformed)}")

    """
    # Print metadata for the first few documents to explore
    num_docs_to_explore = 5  # You can adjust this number
    logger.info(
        f"Exploring metadata for the first {num_docs_to_explore} documents...")
    for doc in docs_from_api[:num_docs_to_explore]:
        logger.info(f"Metadata for document: {doc.metadata}")
    for doc in docs_from_documentation[:num_docs_to_explore]:
        logger.info(f"Metadata for document: {doc.metadata}")
    for doc in docs_from_langsmith[:num_docs_to_explore]:
        logger.info(f"Metadata for document: {doc.metadata}")
    for doc in docs_transformed[:num_docs_to_explore]:
        logger.info(f"Metadata for document: {doc.metadata}")
        """
    # We try to return 'source' and 'title' metadata when querying vector store and
    # Weaviate will error at query time if one of the attributes is missing from a
    # retrieved document.

    logger.info(
        "Ensuring 'source' and 'title' metadata are present in each document...")
    preprocess_metadata(docs_transformed)
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""
    logger.info("Metadata check complete.")

    logger.info("Initializing Chroma vectorstore...")
    # Initialize Chroma vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(CHROMA_COLLECTION_NAME,
                         embeddings, persist_directory="./"+CHROMA_COLLECTION_NAME)
    vectorstore.persist()
    logger.info("Chroma vectorstore initialized.")

    logger.info("Adding documents to Chroma vectorstore...")
    # Add documents to Chroma vectorstore
    document_ids = vectorstore.add_documents(docs_transformed)
    logger.info(f"Number of documents indexed in Chroma: {len(document_ids)}")

    logger.info("Initializing SQL record manager...")
    record_manager = SQLRecordManager(
        f"chroma/{CHROMA_COLLECTION_NAME}", db_url=RECORD_MANAGER_DB_URL
    )
    logger.info("Creating schema for record manager...")
    record_manager.create_schema()
    logger.info("Record manager schema created.")

    logger.info("Starting the indexing process...")
    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE")
                      or "false").lower() == "true",
    )

    logger.info(f"Indexing stats: {indexing_stats}")
    logger.info("Indexing process completed.")


if __name__ == "__main__":
    ingest_docs()
