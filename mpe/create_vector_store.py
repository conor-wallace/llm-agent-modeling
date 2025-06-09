import asyncio
import logging
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import Loader

# Initialize logging
logging.basicConfig(level=logging.INFO)


def extract_trajectory_documents(directory: str) -> list[Document]:
    """
    Extracts trajectory documents from a specified directory.
    Each file in the directory is treated as a separate document.
    """
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)
    return documents


def split_documents(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # chunk size (characters)
        chunk_overlap=chunk_overlap,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    texts = text_splitter.split_documents(documents)

    return texts


def embed_documents(
    documents: list[Document],
    collection_name: str,
    database_directory: str,
    embedding_model: str = "text-embedding-3-large",
    batch_size: int = 4096,
) -> None:
    # Placeholder for embedding logic
    # This function should take the documents and embed them using a suitable model
    embedding_function = OpenAIEmbeddings(model=embedding_model)

    # Initialize the vector store
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=database_directory,
    )

    # Add documents to the vector store in batches
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        # Add the batch of documents to the vector store
        vector_store.add_documents(batch)


if __name__ == "__main__":
    collection_name = "mpe_db"
    database_directory = "./mpe_db"

    trajectory_dir = "trajectories/spread"

    documents = extract_trajectory_documents(trajectory_dir)
    logging.info(f"Number of trajectory documents loaded: {len(documents)}")

    texts = split_documents(documents)
    logging.info(f"Number of documents after splitting: {len(texts)}")

    embed_documents(
        documents=texts,
        collection_name=collection_name,
        database_directory=database_directory,
        embedding_model="text-embedding-3-large",
        batch_size=128,
    )
    logging.info("Embedding documents successfully.")
