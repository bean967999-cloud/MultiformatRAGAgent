"""Minimal OpenAI-compatible chat client for grounded RAG answers."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()


def answer_with_context(question: str, context: str) -> str:
    base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "")
    if not all((base_url, api_key, model)):
        raise ValueError("请配置 LLM_BASE_URL、LLM_API_KEY 和 LLM_MODEL")

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "请仅根据提供的资料回答问题。资料不足时明确说明，不要编造。回答中引用资料编号。"},
                {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
            ],
        },
        timeout=90,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
