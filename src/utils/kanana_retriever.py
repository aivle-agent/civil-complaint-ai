# src/utils/kanana_retriever.py

import os
# import pickle  <-- Removed
from typing import List, Dict, Any, Optional

import numpy as np
import chromadb
from chromadb import PersistentClient

import torch
from transformers import AutoModel, AutoTokenizer

# 환경 변수 또는 기본 경로 설정
# 프로젝트 루트 기준 경로로 설정하는 것이 좋습니다.
DB_DIR = os.environ.get("LAWDB_DIR", "./data/chroma_lawdb_kanana_clustered")
COLLECTION_NAME = "law_corpus"
# CLUSTER_MODEL_PATH = os.environ.get("CLUSTER_MODEL_PATH", "./data/cluster_model.pkl") <-- Removed

# Kanana 임베딩 모델
EMBED_MODEL_NAME = "kakaocorp/kanana-nano-2.1b-embedding"

# 전역 변수로 모델/클라이언트 캐싱 (Lazy Loading 권장하지만 여기서는 모듈 로드 시 초기화 시도)
_embed_tokenizer = None
_embed_model = None
_EMBED_DEVICE = None
# _cluster_model = None <-- Removed
# _CLUSTER_CENTERS = None <-- Removed
_client = None
_collection = None

def _initialize_resources():
    global _embed_tokenizer, _embed_model, _EMBED_DEVICE
    # global _cluster_model, _CLUSTER_CENTERS <-- Removed
    global _client, _collection

    if _embed_model is None:
        print(f"[RAG] Loading Kanana embedding model: {EMBED_MODEL_NAME}")
        _embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, trust_remote_code=True)
        _embed_model = AutoModel.from_pretrained(
            EMBED_MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        _EMBED_DEVICE = _embed_model.device
        print(f"[RAG] Kanana device = {_EMBED_DEVICE}")

    # Clustering model loading removed

    if _client is None:
        if not os.path.exists(DB_DIR):
             print(f"[WARN] ChromaDB directory not found at {DB_DIR}. Retrieval will fail.")
        
        print(f"[RAG] Initializing ChromaDB client at {DB_DIR}")
        _client = chromadb.PersistentClient(path=DB_DIR)
        try:
            _collection = _client.get_collection(COLLECTION_NAME)
            print(f"[RAG] Using collection: {COLLECTION_NAME}")
        except Exception as e:
            print(f"[WARN] Failed to get collection {COLLECTION_NAME}: {e}")


def get_query_embedding(text: str, max_length: int = 512) -> np.ndarray:
    """질문/민원 텍스트를 Kanana 임베딩 벡터로 변환"""
    _initialize_resources()
    if _embed_tokenizer is None or _embed_model is None:
        raise RuntimeError("Embedding model not initialized")

    enc = _embed_tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(_EMBED_DEVICE)

    with torch.no_grad():
        out = _embed_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            pool_mask=enc["attention_mask"],
        )
    # out[0] : (B, D) pooled embedding
    emb = out[0][0].cpu().numpy()  # (D,)
    return emb


# get_top_clusters function removed


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)


def mmr_rerank(
    query_emb: np.ndarray,
    doc_embs: np.ndarray,
    lambda_mult: float = 0.7,
    top_k: int = 5,
) -> List[int]:
    """
    query_emb: (D,)
    doc_embs: (N, D)
    → MMR 점수 기반으로 선택된 doc index 리스트
    """
    N = doc_embs.shape[0]
    if N <= top_k:
        return list(range(N))

    sims_to_query = np.array([
        _cosine_sim(query_emb, doc_embs[i]) for i in range(N)
    ])

    selected: List[int] = []
    candidate = list(range(N))

    first = int(np.argmax(sims_to_query))
    selected.append(first)
    candidate.remove(first)

    while len(selected) < top_k and candidate:
        mmr_scores = []
        for idx in candidate:
            sim_q = sims_to_query[idx]
            sim_div = max(
                _cosine_sim(doc_embs[idx], doc_embs[j]) for j in selected
            )
            score = lambda_mult * sim_q - (1 - lambda_mult) * sim_div
            mmr_scores.append((score, idx))

        mmr_scores.sort(reverse=True, key=lambda x: x[0])
        best_idx = mmr_scores[0][1]
        selected.append(best_idx)
        candidate.remove(best_idx)

    return selected


def retrieve_with_kanana(
    question: str,
    top_k: int = 5,
    # cluster_top_m: int = 2,  <-- Removed
    overfetch: int = 4,
    restrict_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Kanana 기반 RAG 검색 (클러스터링 없이 전체 검색)

    return:
      [
        {
            "doc": <문서 원문>,
            "meta": <메타데이터 dict>,
            "distance": <Chroma distance>,
            "embedding": <np.ndarray (D,)>,
        },
        ...
      ]
    """
    _initialize_resources()
    if _collection is None:
        print("[WARN] Collection not initialized. Returning empty results.")
        return []

    # 1) 쿼리 임베딩
    try:
        q_emb = get_query_embedding(question)
    except Exception as e:
        print(f"[ERROR] Embedding generation failed: {e}")
        return []

    # 2) 클러스터 라우팅 제거됨 -> 전체 검색
    
    where_filter: Optional[Dict[str, Any]] = None
    if restrict_type:
        where_filter = {"type": restrict_type}

    # 3) Chroma
    n_results = top_k * overfetch

    try:
        res = _collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=n_results,
            where=where_filter, # type: ignore
            include=["documents", "metadatas", "distances", "embeddings"],
        )
    except Exception as e:
        print(f"[ERROR] Chroma query failed: {e}")
        return []

    if not res["documents"] or len(res["documents"][0]) == 0:
        return []

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    embs = np.array(res["embeddings"][0])

    # 4) MMR rerank로 최종 top_k 선택
    selected_idxs = mmr_rerank(q_emb, embs, lambda_mult=0.7, top_k=top_k)

    results: List[Dict[str, Any]] = []
    for i in selected_idxs:
        results.append({
            "doc": docs[i],
            "meta": metas[i],
            "distance": dists[i],
            # "embedding": embs[i], # 임베딩은 state에 저장하기엔 너무 크므로 제외하거나 필요시 포함
        })
    return results
