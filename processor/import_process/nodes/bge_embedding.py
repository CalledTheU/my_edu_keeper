import json
from pathlib import Path

from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.config import get_config
from processor.import_process.exceptions import ValidationError
from processor.import_process.state import ImportGraphState
from utils.client.ai_clients import AIClients


class BgeEmbeddingChunksNode(BaseNode):

    name: str = "bge_embedding_node"  # 节点名称
    """
        BGE嵌入节点 
    """
    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        # 1. 参数校验
        validated_chunks, config = self._validate_get_inputs(state)

        # 2. 获取批量嵌入的阈值
        embedding_batch_chunk_size = getattr(config, 'embedding_batch_size', 8)

        # 3. 准备分批嵌入
        total_length = len(validated_chunks)
        final_chunks = []

        for i in range(0, total_length, embedding_batch_chunk_size):
            batch = validated_chunks[i:i + embedding_batch_chunk_size]
            # 拼接要嵌入的内容，向量嵌入，把嵌入的向量注入到 chunk 中
            batch_chunks = self._process_batch_chunks(batch, i, total_length)
            final_chunks.extend(batch_chunks)

        # 4. 更新&返回state
        state['chunks'] = final_chunks

        return state

    def _validate_get_inputs(self, state: ImportGraphState):
        config = get_config()

        chunks = state.get('chunks')
        if not chunks or not isinstance(chunks, list):
            raise ValidationError(f"chunks为空或者无效", self.name)

        self.logger.info(f"嵌入的块数：{len(chunks)}")
        return chunks, config

    def _process_batch_chunks(self, batch, star_index, total_length):
        # 1. 循环处理所有 chunk 的要嵌入的内容拼接
        embedding_contents = []

        for chunk in batch:
            content = chunk.get('content')
            item_name = chunk.get('item_name')
            embedding_content = f"{content}\n{item_name}"
            embedding_contents.append(embedding_content)

        # 2.使用BGE-M3模型进行嵌入
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
            embedding_result = bge_m3_client.encode_documents(embedding_contents)
            if not embedding_result:
                self.logger.info("嵌入结果为空")
                return batch
        except Exception as e:
            self.logger.error(f"嵌入结果异常：{e}")
            return batch

        # 3. 循环处理所有 chunk 的向量以及注入到每一个 chunk 中
        for index, chunk in enumerate(batch):
            # 3.1 获取稠密向量
            dense_vector = embedding_result['dense'][index].tolist()

            # 3.2 解构 csr 矩阵 & 获取稀疏向量
            csr_array = embedding_result['sparse']
            ind_ptr = csr_array.indptr
            start_ind_ptr = ind_ptr[index]
            end_ind_ptr = ind_ptr[index + 1]
            token_id = csr_array.indices[start_ind_ptr:end_ind_ptr].tolist()
            weight = csr_array.data[start_ind_ptr:end_ind_ptr].tolist()
            sparse_vector = dict(zip(token_id, weight))

            # 3.3 注入
            chunk['dense_vector'] = dense_vector
            chunk['sparse_vector'] = sparse_vector

        self.logger.info(f"开始批量处理chunk嵌入:批次{star_index + 1}-{star_index + len(batch)}/{total_length}")
        return batch

if __name__ == '__main__':
    setup_logging()

    input_path = Path(r"D:\pythoncharm\pythonProject\atguigu-shopkeeper\knowledge\test\output\chunks_item_name.json")
    output_path = r"D:\pythoncharm\pythonProject\atguigu-shopkeeper\knowledge\test\output\chunks_vector.json"

    # 1. 读取上游状态
    if not input_path.exists():
        print(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    # 2. 构建模拟的图状态
    state = {
        "chunks": content
    }

    # 3. 触发节点执行
    node_bge_embedding = BgeEmbeddingChunksNode()
    proceed_result = node_bge_embedding(state)

    # 4. 结果落盘
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(proceed_result, f, ensure_ascii=False, indent=4)

    print(f"向量生成测试完成！结果已成功备份至:\n{output_path}")


