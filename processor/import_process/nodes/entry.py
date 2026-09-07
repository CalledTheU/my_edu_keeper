"""

    导入节点:
           判断文件是pdf还是md

"""
import json
from pathlib import Path

from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.exceptions import ValidationError
from processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    name: str = "entry_node"  # 节点名称
    """
        入口节点
    """

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """
        处理入参处理逻辑
        :param state:
        :return:
        """
        # 导入路径
        import_file_path = state.get("import_file_path")
        # 导出路径
        file_dir = state.get("file_dir")

        if import_file_path and file_dir and Path(import_file_path).is_file() and Path(file_dir).is_dir():
            # 导入的文件
            import_file = Path(import_file_path)
            state["source_file"] = import_file.name
            # 获取文件后缀
            import_file_suffix = import_file.suffix.lower()

            if import_file_suffix == ".pdf":
                state["is_pdf_read_enabled"] = True
                state["pdf_path"] = import_file_path
                state["file_title"] = import_file.stem

            elif import_file_suffix == ".md":
                state["is_md_read_enabled"] = True
                state["md_path"] = import_file_path
                state["file_title"] = import_file.stem

            else:
                raise ValidationError(message=f"数据校验失败: 不支持的文件类型: {import_file_suffix}",
                                      node_name=self.name)
        else:
            raise ValidationError(message="数据校验失败: 导入路径或者导出路径不存在", node_name=self.name)

        return state

if __name__ == '__main__':
    setup_logging()
    input_state = {
        "import_file_path": r"E:\文件存储\笔记\项目-掌柜智库\11_尚硅谷AI全能开发技术之项目【掌柜智库】\2.资料\pdf文档\doc\万用表RS-12的使用.pdf",
        "file_dir": r"E:\文件存储\笔记\项目-掌柜智库\11_尚硅谷AI全能开发技术之项目【掌柜智库】\2.资料\pdf文档\output",
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False
    }
    # 实现类
    entry_node = EntryNode()
    # 调用通过父类的__call__实现process的逻辑(可以打印日志)
    node_state = entry_node(input_state)
    # 或者直接调用process方法(没走父类的__call__,所以没有日志)
    # node_state = entry_node.process(input_state)
    # print(json.dumps(node_state,indent=4,ensure_ascii=False))
