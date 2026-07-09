import pandas as pd
import uuid
import numpy as np
import chromadb
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence_transformers is required for embedding generation. "
        "Install it with `pip install sentence-transformers`."
    ) from e

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=500, chunk_overlap=80, separators=None):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

            def split_text(self, text):
                if not text:
                    return []

                text = text.replace("\r\n", "\n").strip()
                if len(text) <= self.chunk_size:
                    return [text]

                pieces = []
                start = 0
                while start < len(text):
                    end = min(len(text), start + self.chunk_size)
                    if end < len(text):
                        split_at = end
                        for sep in self.separators:
                            if sep in {"", " "}:
                                continue
                            idx = text.rfind(sep, start, end)
                            if idx != -1 and idx > start:
                                split_at = idx + len(sep)
                                break
                        if split_at <= start:
                            split_at = end
                        piece = text[start:split_at].strip()
                        if piece:
                            pieces.append(piece)
                        start = max(start + self.chunk_size - self.chunk_overlap, split_at)
                    else:
                        piece = text[start:end].strip()
                        if piece:
                            pieces.append(piece)
                        break
                return pieces


def make_chunk(product_id, text, extra_metadata=None):
    """Arma un chunk con formato estándar, listo para indexar."""
    metadata = {"product_id": product_id}
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "chunk_id": str(uuid.uuid4()),
        "product_id": product_id,
        "text": text,
        "metadata": metadata,
    }


def chunk_by_review(df):
    """Estrategia 1: cada review individual es un chunk.
    Simple y rápida, pero para productos con muchas reviews cortas puede
    generar demasiados chunks poco informativos.
    """
    chunks = []
    for _, row in df.iterrows():
        chunks.append(make_chunk(
            row["ProductId"],
            row["full_text"],
            {"score": row["Score"], "helpfulness_ratio": row["helpfulness_ratio"]},
        ))
    return chunks

def chunk_by_product_aggregate(df, max_chars=1500):
    """Estrategia 2: se concatenan todas las reviews de un producto y se
    corta cada tanto max_chars. Prioriza reviews con más helpfulness, así el
    contenido más útil queda al principio del chunk.
    """
    chunks = []
    for product_id, group in df.groupby("ProductId"):
        group_sorted = group.sort_values("helpfulness_ratio", ascending=False)
        full_text = " ".join(group_sorted["full_text"].tolist())

        # partimos en bloques de max_chars sin cortar palabras a la mitad
        for i in range(0, len(full_text), max_chars):
            piece = full_text[i:i + max_chars]
            chunks.append(make_chunk(
                product_id, piece,
                {"score_mean": group["Score"].mean(), "n_reviews": len(group)},
            ))
    return chunks


def chunk_fixed_size_recursive(df, chunk_size=500, chunk_overlap=80):
    """Estrategia 3: RecursiveCharacterTextSplitter de LangChain sobre el
    texto agregado por producto. Respeta separadores naturales (puntos,
    saltos de línea) antes de cortar a la fuerza.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for product_id, group in df.groupby("ProductId"):
        full_text = " ".join(group["Text"].tolist())
        pieces = splitter.split_text(full_text)
        for piece in pieces:
            chunks.append(make_chunk(
                product_id, piece,
                {"score_mean": group["Score"].mean(), "n_reviews": len(group)},
            ))
    return chunks


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return embedding_model.encode(texts, show_progress_bar=False).tolist()

def build_chroma_collection(chunks, collection_name):
    """Crea una colección Chroma en memoria e indexa los chunks de una
    estrategia. Usamos una colección nueva por estrategia para no mezclar
    resultados.
    """
    client = chromadb.EphemeralClient()
    # si la colección ya existe de una corrida anterior, la recreamos
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    batch_size = 200
    for i in tqdm(range(0, len(chunks), batch_size), desc=f"Indexando {collection_name}"):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embed_texts([c["text"] for c in batch]),
            metadatas=[c["metadata"] for c in batch],
        )
    return collection


def retrieve(collection, question, top_k=5):
    """Devuelve la lista de product_id recuperados, en orden de relevancia,
    sin duplicados (un mismo producto puede tener varios chunks entre los
    top_k).
    """
    query_embedding = embed_texts([question])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    seen = set()
    ordered_product_ids = []
    for metadata in results["metadatas"][0]:
        pid = metadata["product_id"]
        if pid not in seen:
            seen.add(pid)
            ordered_product_ids.append(pid)
    return ordered_product_ids


def recall_at_k(retrieved_ids, relevant_ids):
    """1 si al menos un producto relevante aparece entre los recuperados."""
    return int(any(pid in relevant_ids for pid in retrieved_ids))


def reciprocal_rank(retrieved_ids, relevant_ids):
    """1 / posición del primer producto relevante encontrado, 0 si no aparece."""
    for rank, pid in enumerate(retrieved_ids, start=1):
        if pid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_strategy(collection, golden_set, top_k=5):
    """Corre todo el golden set contra una colección y devuelve el promedio
    de Recall@k y MRR.
    """
    recalls, rrs = [], []
    for item in golden_set:
        retrieved = retrieve(collection, item["question"], top_k=top_k)
        recalls.append(recall_at_k(retrieved, item["relevant_product_ids"]))
        rrs.append(reciprocal_rank(retrieved, item["relevant_product_ids"]))
    return {
        f"recall@{top_k}": np.mean(recalls),
        "mrr": np.mean(rrs),
    }


