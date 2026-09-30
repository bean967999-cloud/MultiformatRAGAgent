"""File ingestion graph: load -> split -> batched embed and Milvus write."""

from langgraph.graph import END, START, StateGraph

from rag.nodes.insert_milvus import insert_node
from rag.nodes.load_document import load_node
from rag.nodes.split_document import split_node
from rag.state import RAGState

builder = StateGraph(RAGState)
builder.add_node("load_node", load_node)
builder.add_node("split_node", split_node)
builder.add_node("insert_node", insert_node)
builder.add_edge(START, "load_node")
builder.add_edge("load_node", "split_node")
builder.add_edge("split_node", "insert_node")
builder.add_edge("insert_node", END)
graph = builder.compile()
