
from pymilvus import MilvusClient, DataType
import json
import hashlib
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class MilvusVectorStore:

    def __init__(
            self,
            uri: str,
            token:str | None = None
    ):
        # 如果配置了 token，就带 token 连接
        if token:
            self.client = MilvusClient(
                uri=uri,
                token=token
            )

        # 本地 Milvus 没有开启认证时，不传 token
        else:
            self.client = MilvusClient(
                uri=uri
            )

    def test_connect(self):

        """
        测试 Milvus 是否连接成功
        """
        collections = self.client.list_collections()

        print("当前 Mlivus Collection:",collections)

    def create_collection(
        self,
        collection_name:str = "multiformat_rag",
        dimension: int = 1024,
    ):
        #先加一个判断，如果集合不存在，才创建
        if self.client.has_collection(collection_name):
            print(f"Collection 已存在：{collection_name}")
            return

        # 创建schema
        #auto_id=True 为什么写在 create_schema()，而不是 id 字段里面？
        # 因为它控制的是这个 Schema 的主键生成策略。
        schema = self.client.create_schema(
            auto_id= True,

            # Schema 中是否允许“动态字段”
            #不允许随意增加 Schema 外的字段
            #数据结构更加严格、可控
            enable_dynamic_collection= False
        )

        #主键
        schema.add_field(
            field_name = "id",
            datatype = DataType.INT64,
            is_primary = True
        )

        #不是越大越好，也不是越小越好
        #content是正文，所以容量预留大；metadata是描述信息，所以容量预留小。max_length是数据库允许的最大长度，不等于实际ChunkSize
        #chunk
        schema.add_field(
            field_name = "content",
            datatype = DataType.VARCHAR,
            max_length = 65535
        )
        #metadata
        schema.add_field(
            field_name = "metadata",
            datatype = DataType.VARCHAR,
            max_length = 2048
        )
        #dense_vector
        schema.add_field(
            field_name = "dense_vector",
            datatype = DataType.FLOAT_VECTOR,
            dim = dimension
        )

        #collection
        self.client.create_collection(
            collection_name = collection_name,
            schema = schema
        )

        print(f"Collection 创建成功：{collection_name}")
        print("=======================================")

    def create_index(
            self,
            collection_name:str = "multiformat_rag"
    ):
        #创建向量索引dense_vector

        index_params = self.client.prepare_index_params()

        #索引配置
        #IVF_FLAT 把大量向量提前分组,查询时不用把所有向量全部扫描一遍
        #L2欧氏距离
        #nlist 是 IVF 的“分组数量”
        index_params.add_index(
            field_name = "dense_vector",
            index_type="IVF_FLAT",
            metric_type = "L2",
            params = {
                "nlist": 128
            }
        )

        #真正创建索引
        #把刚才准备好的索引配置提交给 Milvus，让 Milvus 对指定 Collection 真正执行索引创建
        #add_index() 是添加“索引配置”，create_index() 才是把配置提交给 Milvus、执行创建。
        self.client.create_index(
            collection_name = collection_name,
            index_params = index_params
        )

        logger.info("Index 创建成功：%s.dense_vector", collection_name)

    def ensure_index(self, collection_name: str = "multiformat_rag"):
        """Create the vector index only when the collection has none."""
        if not self.client.list_indexes(collection_name=collection_name):
            self.create_index(collection_name)

    def vector_dimension(self, collection_name: str = "multiformat_rag") -> int:
        description = self.client.describe_collection(collection_name=collection_name)
        for field in description["fields"]:
            if field["name"] == "dense_vector":
                return int(field["params"]["dim"])
        raise ValueError("Collection 缺少 dense_vector 字段")

    def ensure_managed_collection(self, collection_name: str, dimension: int):
        """Create or validate a collection with stable string primary keys."""
        if not self.client.has_collection(collection_name):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_collection=False)
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=2048)
            schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=4096)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
            self.client.create_collection(collection_name=collection_name, schema=schema)

        description = self.client.describe_collection(collection_name=collection_name)
        fields = {field["name"]: field for field in description["fields"]}
        if description["auto_id"] or fields.get("id", {}).get("type") != DataType.VARCHAR or "source" not in fields:
            raise ValueError(f"{collection_name} 不是可重复入库的新版 Collection；请使用其他名称")
        if self.vector_dimension(collection_name) != dimension:
            raise ValueError("Collection 与当前 Embedding 模型维度不一致，请使用新的 Collection 名称")
        self.ensure_index(collection_name)

    @staticmethod
    def _stable_chunk_id(source: str, position: int) -> str:
        return hashlib.sha256(f"{source.casefold()}\0{position}".encode("utf-8")).hexdigest()

    def replace_document(
            self,
            chunks: list[Document],
            dense_vectors: list[list[float]],
            source: str,
            collection_name: str,
    ) -> dict:
        """Compatibility wrapper for callers with precomputed vectors."""
        if not chunks or len(chunks) != len(dense_vectors):
            raise ValueError("chunks 不能为空，且必须与向量一一对应")
        cursor = 0

        def take_vectors(texts: list[str]) -> list[list[float]]:
            nonlocal cursor
            vectors = dense_vectors[cursor:cursor + len(texts)]
            cursor += len(texts)
            return vectors

        return self.replace_document_stream(
            chunks=chunks,
            embed_documents=take_vectors,
            source=source,
            collection_name=collection_name,
            batch_size=100,
        )

    def replace_document_stream(
            self,
            chunks: list[Document],
            embed_documents,
            source: str,
            collection_name: str,
            batch_size: int = 32,
    ) -> dict:
        """Embed and upsert one batch at a time; prune stale IDs after success."""
        if not chunks:
            raise ValueError("chunks 不能为空")
        if not 1 <= batch_size <= 256:
            raise ValueError("batch_size 必须在 1 到 256 之间")
        if any(str(chunk.metadata.get("source")) != source for chunk in chunks):
            raise ValueError("所有 Chunk 必须来自同一个规范化文件路径")

        first = chunks[:batch_size]
        try:
            first_vectors = embed_documents([chunk.page_content for chunk in first])
        except Exception as error:
            raise RuntimeError(f"第 1 批向量化失败：{error}") from error
        if len(first_vectors) != len(first) or not first_vectors:
            raise ValueError("Embedding 返回的向量数量与 Chunk 数量不一致")
        dimension = len(first_vectors[0])
        if dimension < 1 or any(len(vector) != dimension for vector in first_vectors):
            raise ValueError("Embedding 返回了无效的向量维度")
        self.ensure_managed_collection(collection_name, dimension)

        self.load_collection(collection_name)
        expression = f"source == {json.dumps(source, ensure_ascii=False)}"
        iterator = self.client.query_iterator(
            collection_name=collection_name,
            filter=expression,
            output_fields=["id"],
            batch_size=1000,
            consistency_level="Strong",
        )
        old_ids = set()
        try:
            while batch := iterator.next():
                old_ids.update(item["id"] for item in batch)
        finally:
            iterator.close()

        current_ids = set()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            if start == 0:
                vectors = first_vectors
            else:
                try:
                    vectors = embed_documents([chunk.page_content for chunk in batch])
                except Exception as error:
                    raise RuntimeError(f"第 {start // batch_size + 1} 批向量化失败：{error}") from error
            if len(vectors) != len(batch) or any(len(vector) != dimension for vector in vectors):
                raise ValueError(f"第 {start // batch_size + 1} 批 Embedding 数量或维度不正确")
            data = []
            for position, (chunk, vector) in enumerate(zip(batch, vectors), start=start):
                chunk_id = self._stable_chunk_id(source, position)
                current_ids.add(chunk_id)
                data.append({
                    "id": chunk_id,
                    "source": source,
                    "content": chunk.page_content,
                    "metadata": json.dumps(chunk.metadata, ensure_ascii=False),
                    "dense_vector": vector,
                })
            try:
                self.client.upsert(collection_name=collection_name, data=data)
            except Exception as error:
                raise RuntimeError(f"第 {start // batch_size + 1} 批 Milvus 写入失败：{error}") from error
            logger.info("已写入批次 %d：%d 个 Chunk", start // batch_size + 1, len(batch))

        obsolete = list(old_ids - current_ids)
        for start in range(0, len(obsolete), 1000):
            self.client.delete(collection_name=collection_name, ids=obsolete[start:start + 1000])

        return {"upsert_count": len(chunks), "deleted_count": len(obsolete)}


    def insert_documents(
            self,
            chunks: list[Document],
            dense_vectors: list[list[float]],
            collection_name:str = "multiformat_rag"
    ):
        """
        将 chunks 和对应的 dense_vectors 插入 Milvus。
        """
        #数量;保证文本和向量一一对应，不错位
        if len(chunks) != len(dense_vectors):
            raise ValueError(
                f"chunks数量{len(chunks)}"
                f"和dense_vector 数量 {len(dense_vectors)}不一致"
            )

        if not chunks:
            return {"insert_count": 0}

        dimension = self.vector_dimension(collection_name)
        if any(len(vector) != dimension for vector in dense_vectors):
            raise ValueError(f"向量维度必须为 {dimension}，请检查 Embedding 模型")

        #组装数据
        data = []

        for chunk,vector in zip(chunks,dense_vectors):

            item = {
                "content":chunk.page_content,

                #metadata原本是dict
                #这里转成JSON 字符串再存入 VARCHAR
                #ensure_ascii=False时json.dumps中的控制，JSON 序列化时，中文等非 ASCII 字符要不要转换成 \uXXXX 形式
                "metadata": json.dumps(
                    chunk.metadata,
                    ensure_ascii=False
                ),
                "dense_vector": vector
            }

            data.append(item)

        #插入
        result = self.client.insert(
            collection_name = collection_name,
            data = data
        )

        print(
            f"成功插入{len(data)}条数据到Collection:{collection_name}"
        )

        return result



    def load_collection(
            self,
            collection_name: str = "multiformat_rag"
    ):
        """
        将 Collection 加载到 Milvus 的查询内存中，
        使其可以执行 query 和 vector search。
        """

        self.client.load_collection(
            collection_name=collection_name
        )

        logger.info("Collection 加载成功：%s", collection_name)



    def query_documents(
            self,
            collection_name: str = "multiformat_rag",
            limit: int = 5
    ):
        """
        查询 Milvus 中已经入库的数据，
        用于验证 content、metadata 和 dense_vector 是否正确保存。
        """

        # 从指定 Collection 中查询数据
        result = self.client.query(
            collection_name=collection_name,

            # 不设置具体过滤条件，本次只用于查看已经入库的数据
            filter="",

            # 最多返回多少条数据
            limit=limit,

            # 指定查询结果需要返回哪些字段
            output_fields=[
                "id",
                "content",
                "metadata",
                "dense_vector"
            ],
            consistency_level="Strong",
        )

        return result

    def search_documents(
            self,
            query_vector: list[float],
            collection_name: str = "multiformat_rag",
            limit: int = 5,
    ) -> list[Document]:
        if limit < 1:
            raise ValueError("limit 必须大于 0")
        if len(query_vector) != self.vector_dimension(collection_name):
            raise ValueError("查询向量与 Collection 维度不一致")
        self.load_collection(collection_name)
        hits = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field="dense_vector",
            limit=min(limit * 4, 100),
            output_fields=["content", "metadata"],
            search_params={"metric_type": "L2", "params": {"nprobe": 16}},
            consistency_level="Strong",
        )
        documents = []
        seen_content = set()
        for hit in hits[0]:
            entity = hit["entity"]
            if entity["content"] in seen_content:
                continue
            seen_content.add(entity["content"])
            metadata = json.loads(entity["metadata"])
            metadata["distance"] = hit["distance"]
            documents.append(Document(page_content=entity["content"], metadata=metadata))
            if len(documents) == limit:
                break
        return documents
