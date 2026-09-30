"""One-command demonstration using fictional data and an isolated collection."""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from rag.agent import agent_answer
from rag.graph import graph
from rag.nodes.retrieve import retrieve_node
from rag.query_graph import query_graph

DEMO_TEXT = """# 星河知识助手演示资料

本文件只含虚构测试数据。演示项目代号 ORBIT-27。
星河知识助手于 2026 年 10 月 15 日进入演练环境。
如果连续两次健康检查失败，值班人员必须在 15 分钟内完成回滚。
正式生产发布日期没有在本文件中给出。
回答问题时必须根据本文内容引用来源，不得猜测缺失信息。
"""
POSITIVE_QUESTION = "ORBIT-27 的演练日期和回滚时限分别是什么？"
NEGATIVE_QUESTION = "ORBIT-27 的正式生产发布日期是哪一天？"


def run(output_dir: Path, collection: str, retrieval_only: bool = False) -> dict:
    if not collection.startswith("multiformat_rag_demo"):
        raise ValueError("演示 Collection 名称必须以 multiformat_rag_demo 开头")
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = output_dir / "DemoKnowledge.md"
    if fixture.exists() and fixture.read_text(encoding="utf-8") != DEMO_TEXT:
        raise FileExistsError(f"演示文件已有其他内容，请选择新的输出目录：{fixture}")
    fixture.write_text(DEMO_TEXT, encoding="utf-8")

    os.environ["MILVUS_COLLECTION"] = collection
    indexed = graph.invoke({"file_path": fixture})
    documents = retrieve_node({"question": POSITIVE_QUESTION})["retrieved_documents"]
    retrieval_ok = bool(documents) and any(doc.metadata.get("source") == str(fixture.resolve()) for doc in documents)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collection": collection,
        "fixture": str(fixture),
        "indexed_chunks": indexed["inserted_count"],
        "retrieval_ok": retrieval_ok,
        "retrieved_sources": [doc.metadata.get("source") for doc in documents],
    }
    if not retrieval_only:
        positive = query_graph.invoke({"question": POSITIVE_QUESTION})["answer"]
        negative = query_graph.invoke({"question": NEGATIVE_QUESTION})["answer"]
        agent, agent_sources = agent_answer(POSITIVE_QUESTION)
        result.update({
            "answer": positive,
            "negative_answer": negative,
            "agent_answer": agent,
            "agent_source_count": len(agent_sources),
            "positive_fact_ok": bool(re.search(r"2026.{0,8}10.{0,8}15", positive)) and bool(re.search(r"15\s*分钟|十五分钟", positive)),
            "negative_refusal_ok": any(word in negative for word in ("未", "无法", "没有", "资料不足")),
            "agent_fact_ok": bool(re.search(r"2026.{0,8}10.{0,8}15", agent)) and bool(re.search(r"15\s*分钟|十五分钟", agent)) and bool(agent_sources),
        })
    result["passed"] = all(value for key, value in result.items() if key.endswith("_ok"))
    report = output_dir / "DemoResult.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["result_file"] = str(report)
    return result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="运行隔离的多格式 RAG 演示")
    parser.add_argument("--output-dir", type=Path, default=Path("E:/Codex/MultiformatRagDemo"))
    parser.add_argument("--collection", default="multiformat_rag_demo")
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    result = run(args.output_dir, args.collection, args.retrieval_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
