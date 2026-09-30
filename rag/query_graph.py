"""Question -> Milvus retrieval -> grounded LLM answer."""

from langgraph.graph import END, START, StateGraph

from rag.nodes.answer import answer_node
from rag.nodes.retrieve import retrieve_node
from rag.state import RAGState

builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("answer", answer_node)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)
query_graph = builder.compile()
