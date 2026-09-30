"""Command line entry point for indexing and querying the knowledge base."""

import argparse
import logging
import sys
from pathlib import Path

from rag.graph import graph
from rag.query_graph import query_graph


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="多格式 RAG 知识库")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="解析、切分并入库文件").add_argument("file", type=Path)
    commands.add_parser("search", help="只检索相关片段").add_argument("question")
    commands.add_parser("ask", help="检索并让 LLM 根据资料回答").add_argument("question")
    commands.add_parser("agent", help="让 LLM 调用知识库检索工具回答").add_argument("question")
    args = parser.parse_args()

    if args.command == "index":
        if not args.file.is_file():
            parser.error(f"文件不存在：{args.file}")
        result = graph.invoke({"file_path": args.file})
        print(f"已入库 {result['inserted_count']} 个片段")
        return

    if args.command == "search":
        from rag.nodes.retrieve import retrieve_node

        documents = retrieve_node({"question": args.question})["retrieved_documents"]
        for index, doc in enumerate(documents, 1):
            print(f"[{index}] {doc.metadata}\n{doc.page_content}\n")
        return

    if args.command == "agent":
        from rag.agent import agent_answer

        answer, sources = agent_answer(args.question)
        print(answer)
        print("\n检索来源：")
        for index, source in enumerate(sources, 1):
            print(f"[{index}] {source['metadata']}")
        return

    result = query_graph.invoke({"question": args.question})
    print(result["answer"])
    print("\n来源：")
    for index, doc in enumerate(result["retrieved_documents"], 1):
        print(f"[{index}] {doc.metadata}")


if __name__ == "__main__":
    main()
