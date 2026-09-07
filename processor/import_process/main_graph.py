import json

from langgraph.graph.state import CompiledStateGraph, StateGraph

from processor.import_process.nodes.bge_embedding import BgeEmbeddingChunksNode
from processor.import_process.nodes.ducment_split import DocumentSplitNode
from processor.import_process.nodes.entry import EntryNode
from processor.import_process.nodes.import_milvus import ImportMilvusNode
from processor.import_process.nodes.item_name_recognition import ItemNameRecognitionNode
from processor.import_process.nodes.md_img import MarkDownImageNode
from processor.import_process.nodes.pdf_to_md import PdfToMdNode
from langgraph.constants import END, START

from processor.import_process.state import ImportGraphState, create_default_state
from processor.import_process.base import setup_logging

"""
    判断是什么类型的文件，然后进行相应的处理
"""


def import_router(state: ImportGraphState):
    if state.get("is_pdf_read_enabled"):
        return "PDF"
    elif state.get("is_md_read_enabled"):
        return "MD"
    else:
        return "END"


def create_import_graph() -> CompiledStateGraph:
    """
    创建导入流程图

    Returns:
        编译后的 StateGraph 实例

    流程结构:
        entry_node
              │
              ├── (PDF) ──> pdf_to_md_node ──┐
              │                              │
              └── (MD) ─────────────────────>├──> md_img_node
                                              │
                                              v
                                      document_split_node
                                              │
                                              v
                                      item_name_rec_node
                                              │
                                              v
                                        bge_embedding_node
                                              │
                                              v
                                        import_milvus_node
                                              │
                                              v
                                             END
    """
    # 定义状态
    """
       状态已被state.py中的ImportGraphState类封装，
       create_default_state()函数用于创建默认状态。
    """
    # 定义节点
    """
        节点已被base.py中的BaseNode类封装(基类),后续节点继承自BaseNode类即可
    """
    nodes = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode(),
        "md_img_node": MarkDownImageNode(),
        "document_split_node": DocumentSplitNode(),
        "item_name_rec_node": ItemNameRecognitionNode(),
        "bge_embedding_node": BgeEmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode(),
    }

    # 创建图实例
    graph = StateGraph(ImportGraphState)
    # 添加节点
    for key, node in nodes.items():
        graph.add_node(key, node)
    # 或者也可以挨个添加节点
    # graph.add_node("entry_node", EntryNode)

    # 添加边
    graph.set_entry_point("entry_node")  # 设置入口节点
    # 或者也可以使用
    # graph.add_edge(START,"entry_node")
    graph.add_conditional_edges(
        "entry_node",
        import_router,
        {
            "MD": "md_img_node",
            "PDF": "pdf_to_md_node",
            "END": END
        }
    )
    graph.add_edge("pdf_to_md_node", "md_img_node")
    graph.add_edge("md_img_node", "document_split_node")
    graph.add_edge("document_split_node", "item_name_rec_node")
    graph.add_edge("item_name_rec_node", "bge_embedding_node")
    graph.add_edge("bge_embedding_node", "import_milvus_node")
    graph.add_edge("import_milvus_node", END)

    # 编译图
    return graph.compile()


compiled_graph = create_import_graph()


def run_import_graph(**input_state) -> dict:
    """
    便捷函数：运行导入流程

    Args:
        import_file_path: 输入文件路径（PDF 或 MD）
        file_dir: 本地工作目录

    Returns:
        最终状态字典
    """
    # 1. 创建初始状态
    # import_file_path: str  # 导入文件路径
    # file_dir: str  # 导入(出)文件目录
    # state = {
    #     "import_file_path": import_file_path,
    #     "file_dir": file_dir
    # }
    # 使用 create_default_state 方法,把状态字典转换为 ImportGraphState 实例
    init_state = create_default_state(**input_state)

    # 2. 调用 stream 获取每个节点的处理结果
    final_state = None
    for event in compiled_graph.stream(init_state):
        for node_name, state in event.items():
            # print(f"运行节点: {node_name},运行信息: {state}")
            print(f"运行节点: {node_name}")
            final_state = state

    # 生成流程图（ASCII 码艺术）
    # compiled_graph.get_graph().print_ascii()

    return final_state


if __name__ == '__main__':
    setup_logging()
    input_state = {
        "import_file_path": r"E:\文件存储\笔记\项目-掌柜智库\11_尚硅谷AI全能开发技术之项目【掌柜智库】\2.资料\pdf文档\doc\万用表RS-12的使用.pdf",
        "file_dir": r"D:\pythoncharm\pythonProject\atguigu-shopkeeper\knowledge\test\output",
        "is_pdf_read_enabled" : True,
        "is_md_read_enabled" : False
    }
    import_graph = run_import_graph(**input_state)
    print(f"最后一个节点的输出内容:\n{json.dumps(import_graph,indent=4,ensure_ascii=False)}")
