"""Fast offline checks for the ingestion and answer paths."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from pymilvus import DataType

from rag.loaders.factory import LoaderFactory
from rag.agent import agent_answer
from rag.config import index_batch_size
from rag.demo import run as run_demo
from rag.evaluate import answer_checks, source_citation_match
from rag.graph import graph
from rag.nodes.answer import answer_node
from rag.nodes.split_document import split_node
from rag.vectorstores.milvus_client import MilvusVectorStore


class RAGFlowTests(unittest.TestCase):
    def test_loader_suffixes(self):
        self.assertEqual(type(LoaderFactory.get_loader(Path("a.md"))).__name__, "MarkdownLoader")
        with self.assertRaises(ValueError):
            LoaderFactory.get_loader(Path("a.xls"))

    def test_empty_document_rejected_before_embedding(self):
        with self.assertRaisesRegex(ValueError, "没有可索引"):
            split_node({"documents": []})

    def test_no_hits_does_not_call_llm(self):
        with patch("rag.nodes.answer.answer_with_context") as llm:
            result = answer_node({"question": "问题", "retrieved_documents": []})
            self.assertIn("未检索到", result["answer"])
            llm.assert_not_called()

    def test_answer_includes_source(self):
        doc = Document(page_content="证据", metadata={"source": "a.pdf", "page": 2})
        with patch("rag.nodes.answer.answer_with_context", return_value="回答") as llm:
            result = answer_node({"question": "问题", "retrieved_documents": [doc]})
            self.assertEqual(result["answer"], "回答")
            self.assertIn("a.pdf", llm.call_args.args[1])

    def test_search_maps_hits_to_documents(self):
        store = object.__new__(MilvusVectorStore)
        with patch.object(store, "vector_dimension", return_value=3), patch.object(store, "load_collection"):
            hit = {
                "distance": 0.1,
                "entity": {"content": "证据", "metadata": '{"source":"a.pdf"}'},
            }
            store.client = type("Client", (), {"search": lambda self, **kwargs: [[hit, hit]]})()
            result = store.search_documents([0.1, 0.2, 0.3])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].page_content, "证据")
        self.assertEqual(result[0].metadata["source"], "a.pdf")
        self.assertEqual(result[0].metadata["distance"], 0.1)

    def test_agent_calls_search_tool_and_returns_sources(self):
        tool_call = {"id": "call_1", "function": {"name": "search_knowledge_base", "arguments": '{"query":"测试问题"}'}}
        doc = Document(page_content="证据", metadata={"source": "a.pdf"})
        with patch("rag.agent._chat", side_effect=[{"tool_calls": [tool_call]}, {"content": "答案 [1]"}]) as chat, patch(
            "rag.agent.retrieve_node", return_value={"retrieved_documents": [doc]}
        ) as search:
            answer, sources = agent_answer("测试问题")
        self.assertEqual(answer, "答案 [1]")
        self.assertEqual(sources[0]["metadata"]["source"], "a.pdf")
        search.assert_called_once_with({"question": "测试问题"})
        self.assertEqual(chat.call_args_list[0].args[1], "auto")

    def test_index_graph_connects_loader_split_embedding_and_store(self):
        test_dir = Path("E:/Codex") if Path("E:/Codex").is_dir() else None
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", prefix="RagGraphTest", dir=test_dir, delete=False) as handle:
            handle.write("虚构测试文档：知识库演示项目代号 ORBIT-27。")
            fixture = Path(handle.name)
        try:
            with patch("rag.nodes.insert_milvus.EmbeddingFactory.create") as factory, patch(
                "rag.nodes.insert_milvus.MilvusVectorStore"
            ) as store:
                factory.return_value.embed_documents.side_effect = lambda texts: [[0.0] * 1024 for _ in texts]
                store.return_value.replace_document_stream.side_effect = lambda **kwargs: {
                    "upsert_count": len(kwargs["chunks"]), "deleted_count": 0
                }
                result = graph.invoke({"file_path": fixture})
        finally:
            fixture.unlink(missing_ok=True)
        self.assertGreater(len(result["chunks"]), 0)
        self.assertEqual(result["inserted_count"], len(result["chunks"]))
        store.return_value.replace_document_stream.assert_called_once()
        self.assertNotIn("dense_vectors", result)

    def test_replace_document_removes_obsolete_chunks_after_upsert(self):
        store = object.__new__(MilvusVectorStore)
        store.client = MagicMock()
        source = r"E:\Codex\RagSample.txt"
        current_id = store._stable_chunk_id(source, 0)
        old_id = store._stable_chunk_id(source, 1)
        store.client.query_iterator.return_value.next.side_effect = [[{"id": current_id}, {"id": old_id}], []]
        chunk = Document(page_content="新内容", metadata={"source": source})
        with patch.object(store, "ensure_managed_collection"), patch.object(store, "load_collection"):
            result = store.replace_document([chunk], [[0.1, 0.2, 0.3]], source, "v2")
        self.assertEqual(result, {"upsert_count": 1, "deleted_count": 1})
        self.assertEqual(store.client.upsert.call_args.kwargs["data"][0]["id"], current_id)
        store.client.delete.assert_called_once_with(collection_name="v2", ids=[old_id])

    def test_failed_upsert_preserves_old_chunks(self):
        store = object.__new__(MilvusVectorStore)
        store.client = MagicMock()
        source = r"E:\Codex\RagSample.txt"
        store.client.query_iterator.return_value.next.side_effect = [[{"id": "old"}], []]
        store.client.upsert.side_effect = RuntimeError("write failed")
        chunk = Document(page_content="新内容", metadata={"source": source})
        with patch.object(store, "ensure_managed_collection"), patch.object(store, "load_collection"):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                store.replace_document([chunk], [[0.1, 0.2, 0.3]], source, "v2")
        store.client.delete.assert_not_called()

    def test_streaming_embedding_uses_bounded_batches(self):
        store = object.__new__(MilvusVectorStore)
        store.client = MagicMock()
        store.client.query_iterator.return_value.next.return_value = []
        source = r"E:\Codex\RagSample.txt"
        chunks = [Document(page_content=f"part {i}", metadata={"source": source}) for i in range(5)]
        embed = MagicMock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])
        with patch.object(store, "ensure_managed_collection"), patch.object(store, "load_collection"):
            result = store.replace_document_stream(chunks, embed, source, "v2", batch_size=2)
        self.assertEqual(result["upsert_count"], 5)
        self.assertEqual([len(call.args[0]) for call in embed.call_args_list], [2, 2, 1])
        self.assertEqual([len(call.kwargs["data"]) for call in store.client.upsert.call_args_list], [2, 2, 1])
        store.client.delete.assert_not_called()

    def test_invalid_batch_size_rejected(self):
        with patch.dict("os.environ", {"INDEX_BATCH_SIZE": "0"}):
            with self.assertRaisesRegex(ValueError, "INDEX_BATCH_SIZE"):
                index_batch_size()

    def test_second_batch_failure_does_not_prune_old_chunks(self):
        store = object.__new__(MilvusVectorStore)
        store.client = MagicMock()
        store.client.query_iterator.return_value.next.side_effect = [[{"id": "old"}], []]
        store.client.upsert.side_effect = [None, RuntimeError("Milvus unavailable")]
        source = r"E:\Codex\RagSample.txt"
        chunks = [Document(page_content=f"part {i}", metadata={"source": source}) for i in range(2)]
        with patch.object(store, "ensure_managed_collection"), patch.object(store, "load_collection"):
            with self.assertRaisesRegex(RuntimeError, "第 2 批 Milvus 写入失败"):
                store.replace_document_stream(chunks, lambda texts: [[0.1, 0.2, 0.3] for _ in texts], source, "v2", 1)
        store.client.delete.assert_not_called()

    def test_legacy_collection_rejected_for_indexing(self):
        store = object.__new__(MilvusVectorStore)
        store.client = MagicMock()
        store.client.has_collection.return_value = True
        store.client.describe_collection.return_value = {
            "auto_id": True,
            "fields": [{"name": "id", "type": DataType.INT64}],
        }
        with self.assertRaisesRegex(ValueError, "不是可重复入库"):
            store.ensure_managed_collection("multiformat_rag", 1024)
        store.client.create_collection.assert_not_called()

    def test_evaluation_recognizes_supported_refusal_wording(self):
        case = {"answerable": False, "required_patterns": []}
        result = answer_checks(case, "AURORA-52 没有指定任何加密算法。[1]", 2)
        self.assertTrue(result["refusal"])
        self.assertTrue(result["citation_valid"])

    def test_evaluation_citation_must_point_to_expected_file(self):
        self.assertTrue(source_citation_match("依据[2]。", ["wrong.txt", "right.pdf"], "right.pdf"))
        self.assertFalse(source_citation_match("依据[1]。", ["wrong.txt", "right.pdf"], "right.pdf"))

    def test_demo_refuses_to_write_to_non_demo_collection(self):
        with self.assertRaisesRegex(ValueError, "演示 Collection"):
            run_demo(Path("E:/Codex/UncreatedDemoTest"), "multiformat_rag_v2", retrieval_only=True)


if __name__ == "__main__":
    unittest.main()
