from typing import Dict

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .prompts import QUERY_PROMPT
    

def create_document(agent_fingerprints_0: Dict[str, float], teammate_type: str):
    query_prompt = QUERY_PROMPT.format(
        cumulative_reward=f"{agent_fingerprints_0["cumulative_reward"]:.2f}",
        dwell_onion=f"{agent_fingerprints_0["dwell_onion"]:.2f}",
        dwell_plate=f"{agent_fingerprints_0["dwell_plate"]:.2f}",
        dwell_pot=f"{agent_fingerprints_0["dwell_pot"]:.2f}",
        dwell_window=f"{agent_fingerprints_0["dwell_window"]:.2f}",
        near_onion_pile_steps=f"{agent_fingerprints_0["near_onion_pile_steps"]:.2f}",
        near_plate_pile_steps=f"{agent_fingerprints_0["near_plate_pile_steps"]:.2f}",
        near_pot_steps=f"{agent_fingerprints_0["near_pot_steps"]:.2f}",
        near_window_steps=f"{agent_fingerprints_0["near_window_steps"]:.2f}"
    )

    return Document(
        page_content=query_prompt,
        metadata={
            "teammate_type": teammate_type
        }
    )


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