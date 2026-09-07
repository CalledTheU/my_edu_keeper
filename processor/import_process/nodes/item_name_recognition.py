import json
import os
from typing import Tuple, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.exceptions import StateFieldError, ValidationError
from processor.import_process.state import ImportGraphState
from prompt.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
from utils.client.ai_clients import AIClients
from utils.client.storage_clients import StorageClients


class ItemNameRecognitionNode(BaseNode):
    name: str = "item_name_recognition_node"  # 节点名称
    """
        商品名称识别节点 
    """

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """商品名识别节点处理流程"""

        # 1. 参数校验
        file_title, chunks, item_name_chunks_k, item_name_chunk_size = self._validate_state(state)

        # 2. 构建商品名识别上下文
        item_name_recognition_context: str = self._prepare_item_name_recognition_context(
            chunks, item_name_chunks_k, item_name_chunk_size
        )

        # 3. LLM商品名识别
        item_name = self._recognition_name(file_title, item_name_recognition_context)

        # 4. 向量化提取到商品名
        dense_vector: list[float] = []
        sparse_vector: Dict[int, float]
        dense_vector, sparse_vector = self._embedding_item_name(item_name)

        # 5. 存储到milvus中
        self._insert_milvus(file_title, item_name, dense_vector, sparse_vector,
                            self.config.item_name_collection)

        # 6. 回填item_name信息
        self._fill_item_name(item_name, state, chunks)

        # 7.备份，给下个节点准备下测试数据。
        self._backup_chunks(state, chunks)

        return state

    def _validate_state(self, state):
        # 获取state中的文件标题和分块内容
        file_title = state.get('file_title')
        chunks = state.get('chunks')

        # 校验文件标题
        if not file_title:
            raise StateFieldError(node_name=self.name, field_name="file_title", expected_type=str)
        # 校验分块内容
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)

        # 从配置文件中获取商品名分块数量(3块)
        item_name_chunks_k = self.config.item_name_chunk_k
        if not item_name_chunks_k or item_name_chunks_k <= 0:
            raise ValidationError(message="item_name_chunk_k为空或者无效", node_name=self.name)

        # 从配置文件中获取商品名分块大小(2500)
        item_name_chunk_size = self.config.item_name_chunk_size
        if not item_name_chunk_size or item_name_chunk_size <= 0:
            raise ValidationError(message="item_name_chunk_size为空或者无效", node_name=self.name)

        # 返回文件标题、分块内容、商品名分块数量、商品名分块大小
        return file_title, chunks, item_name_chunks_k, item_name_chunk_size

    def _prepare_item_name_recognition_context(self, chunks, item_name_chunks_k, item_name_chunk_size):
        final_chunk_text = []
        for index, chunk in enumerate(chunks[:item_name_chunks_k]):
            # 判断分块内容是否为空
            if not chunk:
                continue
            # 获取分块中的正文内容
            content = chunk.get("content")
            # 拼接分块内容
            chunk_text = f"切片{index + 1}: {content}"

            # 判断拼接后的分块内容是否超过商品名分块大小
            if len(final_chunk_text) + len(chunk_text) > item_name_chunk_size:
                # 如果超过，则跳出循环
                break
            # 放入最终的集合
            final_chunk_text.append(chunk_text)
        # 返回最终的分块内容
        return "\n".join(final_chunk_text)

    def _recognition_name(self, file_title, item_name_recognition_context):
        try:
            # 创建大模型
            llm = AIClients.get_deepseek_llm(response_format=False)

            # 构建用户提示词
            human_message = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title,
                                                                  context=item_name_recognition_context)

            # 让大模型对上下文生成商品名
            ai_message = llm.invoke(
                [SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT), HumanMessage(content=human_message)])

            # 返回大模型生成的结果
            ai_message_content = ai_message.content.strip()
            if not ai_message_content or ai_message_content == "UNKNOWN":
                self.logger.info(f"LLM未识别出商品名，降级使用标题: {file_title}")
                return file_title

            return json.loads(ai_message_content).get("item_name")
        except Exception as e:
            self.logger.error(f"LLM调用失败，降级使用标题: {file_title}，异常: {e}")
            return file_title

    def _embedding_item_name(self, item_name) -> Tuple[List[float], Dict[int, float]]:
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
            vector_result = bge_m3_client.encode_documents([item_name])

            dense_vector = vector_result['dense'][0].tolist()
            start_index = vector_result['sparse'].indptr[0]
            end_index = vector_result['sparse'].indptr[1]
            token_id = vector_result['sparse'].indices[start_index:end_index].tolist()
            weight = vector_result['sparse'].data[start_index:end_index].tolist()
            sparse_vector = dict(zip(token_id, weight))

            return dense_vector, sparse_vector
        except ConnectionError as e:
            self.logger.error(f"BGE-M3 客户端获取失败: {e}")
            return None, None
        except Exception as e:
            self.logger.error(f"商品名 [{item_name}] 向量化处理失败: {e}")
            return None, None

    def _insert_milvus(self, file_title, item_name, dense_vector, sparse_vector, item_name_collection):
        if not dense_vector or not sparse_vector:
            self.logger.error(f"文档{file_title} 对应的商品名{item_name} 向量生成不完整")
            return

        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"Milvus 客户端创建失败: {e}")
            return

        try:
            if not milvus_client.has_collection(item_name_collection):
                self._create_item_name_collection(item_name_collection, milvus_client)

            data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
            result = milvus_client.insert(collection_name=item_name_collection, data=[data])
            self.logger.info(f"已成功保存到 Milvus，ID: {result['ids'][0]}")
        except Exception as e:
            self.logger.error(f"Milvus 数据操作失败: {e}")

    def _create_item_name_collection(self, item_name_collection, milvus_client):
        schema = milvus_client.create_schema()

        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_param = milvus_client.prepare_index_params()
        index_param.add_index(field_name="dense_vector", index_name="dense_vector_index",
                              index_type="AUTOINDEX", metric_type="COSINE")
        index_param.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                              index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        milvus_client.create_collection(collection_name=item_name_collection, schema=schema, index_params=index_param)
        self.logger.info(f"集合 {item_name_collection} 创建成功并构建了索引")

    def _fill_item_name(self, item_name, state, chunks):
        for chunk in chunks:
            chunk['item_name'] = item_name
        state['item_name'] = item_name

    def _backup_chunks(self, state, chunks):
        """
        将回填item_name的切片列表结果备份到json文件，给下个节点单元测试使用。
        :param state:
        :param chunks:
        :return:
        """
        local_dir = state.get("file_dir", "")
        os.makedirs(local_dir, exist_ok=True) #exist_ok=True：如果目录已存在，不会抛出异常，直接跳过
        output_path = os.path.join(local_dir, "chunks_item_name.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.warning(f"备份失败：{e}")

if __name__ == '__main__':
    setup_logging()

    # 1. 读取chunk.json
    chunk_json_path = r"D:\pythoncharm\pythonProject\atguigu-shopkeeper\knowledge\test\output\chunks.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    # 2. 构建state
    state = {
        "file_dir": r"D:\pythoncharm\pythonProject\atguigu-shopkeeper\knowledge\test\output",
        "file_title": "万用表RS-12的使用",
        "chunks": chunk_content
    }

    # 3. 实例化节点
    node = ItemNameRecognitionNode()

    # 4. 调用process
    result = node(state)

    # # 5. 输出结果
    # print(f"商品名: {result.get('item_name')}")
    # print(f"chunks数量: {len(result.get('chunks', []))}")
    # print(f"首个chunk是否含item_name: {'item_name' in result['chunks'][0]}")
    from rich import print as uprint
    uprint(result)