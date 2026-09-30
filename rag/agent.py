"""A bounded tool-calling agent backed by knowledge-base search."""

import json
import os

import requests
from dotenv import load_dotenv

from rag.nodes.retrieve import retrieve_node

load_dotenv()

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Search indexed documents for evidence relevant to a question.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search question or keywords"}},
            "required": ["query"],
        },
    },
}


def _chat(messages, tool_choice):
    base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "")
    if not all((base_url, api_key, model)):
        raise ValueError("请配置 LLM_BASE_URL、LLM_API_KEY 和 LLM_MODEL")
    payload = {
        "model": model,
        "temperature": 0,
        "messages": messages,
    }
    if tool_choice != "none":
        payload.update({"tools": [SEARCH_TOOL], "tool_choice": "auto"})
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


def agent_answer(question: str) -> tuple[str, list[dict]]:
    """Require evidence retrieval, then permit up to two follow-up tool calls."""
    if not question.strip():
        raise ValueError("问题不能为空")
    messages = [
        {"role": "system", "content": "你是知识库助手。先搜索知识库，仅根据工具返回的资料回答。资料不足时明确说明，不要编造。引用来源编号。"},
        {"role": "user", "content": question},
    ]
    sources = []
    seen_content = set()
    for attempt in range(3):
        message = _chat(messages, "auto")
        calls = message.get("tool_calls") or []
        if not calls:
            if not sources:
                return "未检索到相关资料，无法根据知识库回答。", sources
            return message.get("content") or "资料不足，无法回答。", sources

        messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": calls})
        for call in calls:
            if call["function"]["name"] != "search_knowledge_base":
                raise ValueError("模型请求了未知工具")
            arguments = json.loads(call["function"]["arguments"])
            query = arguments.get("query", "").strip()
            if not query:
                raise ValueError("检索词不能为空")
            documents = retrieve_node({"question": query})["retrieved_documents"]
            results = []
            for doc in documents:
                if doc.page_content in seen_content:
                    continue
                seen_content.add(doc.page_content)
                source = {"content": doc.page_content, "metadata": doc.metadata}
                sources.append(source)
                results.append({"number": len(sources), **source})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(results, ensure_ascii=False)})

    # A final call must produce an answer from the collected tool results.
    if not sources:
        return "未检索到相关资料，无法根据知识库回答。", sources
    messages.append({"role": "system", "content": "停止调用工具。根据已有检索结果回答；资料不足就说明无法回答。"})
    final = _chat(messages, "none")
    return final.get("content") or "资料不足，无法回答。", sources
