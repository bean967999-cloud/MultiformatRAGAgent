"""Shared runtime configuration for Milvus collections."""

import os
import re


DEFAULT_COLLECTION = "multiformat_rag_v2"


def collection_name() -> str:
    name = (os.getenv("MILVUS_COLLECTION") or DEFAULT_COLLECTION).strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,254}", name):
        raise ValueError("MILVUS_COLLECTION 必须是有效的 Milvus Collection 名称")
    return name


def index_batch_size() -> int:
    raw = os.getenv("INDEX_BATCH_SIZE", "32")
    try:
        size = int(raw)
    except ValueError as error:
        raise ValueError("INDEX_BATCH_SIZE 必须是 1 到 256 的整数") from error
    if not 1 <= size <= 256:
        raise ValueError("INDEX_BATCH_SIZE 必须是 1 到 256 的整数")
    return size
