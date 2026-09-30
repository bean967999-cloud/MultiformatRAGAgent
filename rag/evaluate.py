"""Run a small, reproducible retrieval and answer evaluation set."""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from rag.config import collection_name
from rag.embeddings.factory import EmbeddingFactory
from rag.nodes.answer import answer_node
from rag.vectorstores.milvus_client import MilvusVectorStore

load_dotenv()


def cited_numbers(answer: str) -> set[int]:
    return {int(number) for number in re.findall(r"(?:\[|【)(\d+)(?:\]|】)", answer)}


def source_citation_match(answer: str, sources: list[str], expected_source: str) -> bool:
    return any(sources[number - 1] == expected_source for number in cited_numbers(answer) if 1 <= number <= len(sources))


def answer_checks(case: dict, answer: str, document_count: int) -> dict:
    citations = cited_numbers(answer)
    citation_valid = bool(citations) and all(1 <= number <= document_count for number in citations)
    if case["answerable"]:
        facts_match = all(re.search(pattern, answer, re.IGNORECASE) for pattern in case["required_patterns"])
        return {"facts_match": bool(facts_match), "citation_valid": citation_valid, "refusal": None}
    refusal = bool(re.search(r"资料不足|无法确认|无法确定|未提供|未指定|未提及|没有指定|没有提供|没有.*资料|cannot|not specified", answer, re.IGNORECASE))
    return {"facts_match": None, "citation_valid": citation_valid, "refusal": refusal}


def summarize(rows: list[dict], cases: list[dict]) -> dict:
    positives = [row for row, case in zip(rows, cases) if case["answerable"]]
    negatives = [row for row, case in zip(rows, cases) if not case["answerable"]]
    metric = lambda group, key: {"passed": sum(row.get(key) is True for row in group), "total": len(group)}
    return {
        "retrieval_hit_at_1": metric(positives, "hit_at_1"),
        "retrieval_hit_at_3": metric(positives, "hit_at_3"),
        "retrieval_hit_at_5": metric(positives, "hit_at_5"),
        "answer_facts": metric(positives, "facts_match"),
        "answer_citations": metric(positives, "citation_valid"),
        "answer_source_citations": metric(positives, "citation_source_match"),
        "negative_refusals": metric(negatives, "refusal"),
    }


def evaluate(cases: list[dict], collection: str) -> dict:
    uri = os.getenv("MILVUS_URI")
    if not uri:
        raise ValueError("请配置 MILVUS_URI")
    embedding = EmbeddingFactory.create()
    store = MilvusVectorStore(uri=uri, token=os.getenv("MILVUS_TOKEN"))
    rows = []
    for case in cases:
        question = case["question"]
        row = {"id": case["id"], "question": question, "expected_source": case["expected_source"]}
        try:
            documents = store.search_documents(embedding.embed_query(question), collection_name=collection, limit=5)
            sources = [Path(document.metadata.get("source", "")).name for document in documents]
            row["retrieved_sources"] = sources
            if case["answerable"]:
                row["hit_at_1"] = case["expected_source"] in sources[:1]
                row["hit_at_3"] = case["expected_source"] in sources[:3]
                row["hit_at_5"] = case["expected_source"] in sources[:5]
            answer = answer_node({"question": question, "retrieved_documents": documents})["answer"]
            row["answer"] = answer
            row.update(answer_checks(case, answer, len(documents)))
            if case["answerable"]:
                row["citation_source_match"] = source_citation_match(answer, sources, case["expected_source"])
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collection": collection,
        "case_count": len(cases),
        "metrics": summarize(rows, cases),
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="评估 RAG 检索和回答")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rescore-existing", action="store_true", help="只重算已有回答的评分，不重新调用模型")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.rescore_existing:
        result = json.loads(args.output.read_text(encoding="utf-8"))
        by_id = {row["id"]: row for row in result["cases"]}
        if set(by_id) != {case["id"] for case in cases}:
            raise ValueError("已有结果与当前题集不一致")
        rows = [by_id[case["id"]] for case in cases]
        for row, case in zip(rows, cases):
            if "answer" in row:
                row.update(answer_checks(case, row["answer"], len(row["retrieved_sources"])))
                if case["answerable"]:
                    row["citation_source_match"] = source_citation_match(
                        row["answer"], row["retrieved_sources"], case["expected_source"]
                    )
        result["cases"] = rows
        result["metrics"] = summarize(rows, cases)
        result["rescored_at"] = datetime.now(timezone.utc).isoformat()
    else:
        result = evaluate(cases, collection_name())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"详细结果：{args.output}")


if __name__ == "__main__":
    main()
