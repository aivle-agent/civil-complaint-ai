import os
import re
import json
import pandas as pd
import numpy as np
from typing import Any, List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN



def sanitize_text(x: Any) -> str:
    if not isinstance(x, str):
        return ""

    text = x
    text = re.sub(r"\d{2,3}-\d{3,4}-\d{4}", "[TEL]", text)
    text = re.sub(r"\d{6}-\d{7}", "[RRN]", text)
    text = re.sub(r"[A-Za-z0-9\._%+-]+@[A-Za-z0-9\.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    text = re.sub(r"\d{10,}", "[NUMSEQ]", text)
    text = re.sub(r"[가-힣]{2,3}씨", "[NAME]", text)
    return text.strip()

# 2. 지방·중앙 행정기관 데이터 파싱 (Q/A 구분)


def parse_question_answer(full_text: str) -> Tuple[str, str]:
    if not isinstance(full_text, str):
        return "", ""

    text = full_text.strip()

    a_idx = (
        text.find("\nA :") if "\nA :" in text
        else text.find("A :") if "A :" in text
        else -1
    )

    if a_idx != -1:
        q_part = text[:a_idx].strip()
        a_part = text[a_idx:].strip()
        a_part = re.sub(r"^A\s*:\s*", "", a_part).strip()
    else:
        q_part, a_part = text, ""

    q_part = re.sub(r"^제목\s*:\s*", "", q_part)
    q_part = re.sub(r"^Q\s*:\s*", "", q_part)

    return sanitize_text(q_part), sanitize_text(a_part)


def load_qa_from_consulting_csv(csv_path: str, text_col="consulting_content") -> List[Dict[str, str]]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if text_col not in df.columns:
        raise ValueError(f"{csv_path}에 {text_col} 컬럼 없음")

    data = []
    for _, row in df.iterrows():
        q, a = parse_question_answer(row[text_col])
        if len(q) >= 5 and len(a) >= 5:
            data.append({"question": q, "answer": a})

    print(f"[LOAD] {csv_path} → {len(data)} Q/A")
    return data


# 3. 국민신문고 데이터 (question only)


def load_novel_test_from_sinmungo(csv_path: str, text_col="content_full") -> List[Dict[str, str]]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if text_col not in df.columns:
        raise ValueError(f"{csv_path}에 {text_col} 컬럼 없음")

    data = []
    for _, row in df.iterrows():
        q = sanitize_text(row[text_col])
        if len(q) >= 20:
            data.append({"question": q, "answer": ""})

    print(f"[LOAD] {csv_path} → {len(data)} items (novel test)")
    return data


# 4. Train / Test 데이터 구성


def build_train_and_novel_sets_raw():
    train_files = [
        ("data/지방행정기관.csv", "consulting_content"),
        ("data/중앙행정기관.csv", "consulting_content"),
    ]

    train_data = []
    for path, col in train_files:
        train_data.extend(load_qa_from_consulting_csv(path, col))

    novel_test_data = load_novel_test_from_sinmungo("data/국민신문고.csv")

    return train_data, novel_test_data

# 5. Deduplication


def deduplicate_train(train_data):
    df = pd.DataFrame(train_data)
    df["pair_count"] = df.groupby(["question", "answer"])["answer"].transform("count")
    df_unique = df.drop_duplicates(["question", "answer"]).reset_index(drop=True)
    print(f"[DEDUP] Train {len(df)} → {len(df_unique)} unique")
    return df_unique.to_dict(orient="records")


def deduplicate_novel(novel):
    df = pd.DataFrame(novel)
    df["q_count"] = df.groupby("question")["question"].transform("count")
    df_unique = df.drop_duplicates("question").reset_index(drop=True)
    print(f"[DEDUP] Novel {len(df)} → {len(df_unique)} unique")
    return df_unique.to_dict(orient="records")


# 6. 클러스터링


def cluster_train_questions(train_data, eps=0.3, min_samples=5):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    questions = [d["question"] for d in train_data]
    embeddings = model.encode(questions, convert_to_numpy=True, show_progress_bar=True)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = db.fit_predict(embeddings)

    for d, lab in zip(train_data, labels):
        d["cluster_id"] = int(lab)

    print(f"[CLUSTER] clusters = {len(set(labels)) - (1 if -1 in labels else 0)}")
    print(f"[CLUSTER] noise ratio = {(labels == -1).mean():.2%}")

    return train_data, labels

# 7. JSONL Export


def export_to_jsonl(train, novel, out_dir="export"):
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train_data.jsonl")
    novel_path = os.path.join(out_dir, "novel_test.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for row in train:
            f.write(json.dumps({
                "input": row["question"],
                "output": row["answer"],
                "cluster_id": int(row.get("cluster_id", -1)),
                "pair_count": int(row.get("pair_count", 1)),
            }, ensure_ascii=False) + "\n")

    with open(novel_path, "w", encoding="utf-8") as f:
        for row in novel:
            f.write(json.dumps({
                "input": row["question"],
                "output": "",
                "q_count": int(row.get("q_count", 1)),
            }, ensure_ascii=False) + "\n")

    print(f"\n[EXPORT DONE]")
    print(f"Train JSONL → {train_path}")
    print(f"Novel JSONL → {novel_path}")

# 8. MAIN 실행


if __name__ == "__main__":
    train_raw, novel_raw = build_train_and_novel_sets_raw()

    train_unique = deduplicate_train(train_raw)
    novel_unique = deduplicate_novel(novel_raw)

    train_clustered, labels = cluster_train_questions(train_unique)

    export_to_jsonl(train_clustered, novel_unique, out_dir="export")
