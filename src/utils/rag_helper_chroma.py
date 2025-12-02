import os
from typing import List
from functools import lru_cache
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

"""
프로젝트 내 벡터DB가 있으면 해당 벡터DB를 참조하고,
벡터DB가 없으면 임베딩 및 구축합니다.
"""

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
VECTOR_DIR = os.path.join(DATA_DIR, "vectorstores", "civil_chroma")


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceBgeEmbeddings:
    """
    임베딩 모델: sentence-transformers/all-MiniLM-L6-v2
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceBgeEmbeddings(
        model_name = model_name,
        model_kwargs = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )
    return embeddings



@lru_cache(maxsize=1)
def load_to_document() -> List[Document]:
    '''
    csv 파일 불러와서 Document화
    '''
    docs: List[Document] = []

    csv_configs = [
        {
            "filename": "중앙행정기관.csv",
            "source_column": "consulting_content"
        },
        {
            "filename": "지방행정기관.csv",
            "source_column": "consulting_content"
        },
        {
            "filename": "국민신문고.csv",
            "source_column": "content_problem"
        },
    ]

    for cfg in csv_configs:
        csv_path = os.path.join(DATA_DIR, cfg["filename"])
        if not os.path.exists(csv_path):
            print(f"[RAG] Warning: {csv_path} not found. skip")
            continue
        
        try:
            csv_loader = CSVLoader(file_path=csv_path, 
                                   encoding="utf-8-sig", 
                                   source_column=cfg["source_column"])
            file_docs = csv_loader.load()

            # 메타데이터에 파일 이름 추가
            for d in file_docs:
                d.metadata = d.metadata or {}
                d.metadata["source_file"] = cfg["filename"]
            
            docs.extend(file_docs)
            print(f"[RAG] {len(file_docs)}개 docs 로드됨. from {cfg['filename']}")
        
        except Exception as e:
            print(f"[RAG] Error loading {cfg['filename']}: {e}")
    
    print(f"[RAG] {len(docs)} documents 로드 완료. from CSVs (via CSVLoader).")
    return docs



@lru_cache(maxsize=1)
def get_civil_vectorstore() -> Chroma:
    """
    벡터스토어 & 리트리버
    """
    embeddings = get_embedding_model()

    # 이미 벡터DB가 존재하면 로드
    # os.listdir(VECTOR_DIR): 해당 디렉토리 내 파일들을 리스트로 반환
    if os.path.exists(VECTOR_DIR) and os.listdir(VECTOR_DIR):
        print(f"[RAG] 이미 {VECTOR_DIR}에 존재하는 Chroma DB 로딩")
        vectorstore = Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=embeddings
        )
        return vectorstore
    
    # 벡터DB가 존재하지 않으면 생성.
    os.makedirs(VECTOR_DIR, exist_ok=True)
    print(f"[RAG] {VECTOR_DIR}에 새로운 Chroma DB 생성")
    docs = load_to_document()
    if not docs:
        raise RuntimeError("[RAG] Document 로드 안됨. data/ 에서 CSV 파일을 확인하세요.")
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTOR_DIR
    )
    vectorstore.persist()
    print("[RAG] Chroma DB 구축 및 저장(persist) 완료")
    return vectorstore



@lru_cache(maxsize=1)
def get_civil_retriever() -> VectorStoreRetriever:
    vectorstore = get_civil_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": 3}
    )
    return retriever