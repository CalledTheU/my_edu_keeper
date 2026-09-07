import subprocess
from pathlib import Path

from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.exceptions import ValidationError, FileProcessingError, PdfConversionError
from processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    name: str = "pdf_to_md_node"  # 节点名称
    """
        PDF转MD节点 
    """

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """
            pdf转md处理逻辑
        """
        # 校验输入输出路径
        import_file_obj, file_dir_obj = self._validate_state_inputs_path(state)
        # 执行MinerU转换
        execute_mineru = self._execute_mineru(import_file_obj, file_dir_obj)
        # 判断转换是否成功
        if execute_mineru != 0:
            raise PdfConversionError("PDF转换失败", self.name)
        # 获取MD路径
        md_path = self._get_md_path(import_file_obj, file_dir_obj)
        # 更新state，添加md_path键值对
        state["md_path"] = str(md_path)
        return state

    """
        校验输入的文件 & 输出路径 是否存在
    """
    def _validate_state_inputs_path(self, state) -> tuple[Path, Path]:
        # 导入文件路径
        import_file_path = state.get("import_file_path")

        # 导出文件目录
        file_dir = state.get("file_dir")

        # PDF文件路径
        pdf_path = state.get("pdf_path")

        # 验证导入文件路径
        if not import_file_path:
            raise ValidationError("import_file_path参数不存在", self.name)

        # 转换为Path对象
        import_file_obj = Path(import_file_path)

        # 验证文件存在
        if not import_file_obj.exists():
            raise FileProcessingError(f"import_file_path文件不存在: {import_file_path}", self.name)

        # 验证导出文件目录
        if not file_dir:
            # 此处可进行降级处理:使用默认目录,或者创建一个目录用来保存转换后的MD文件
            raise ValidationError("file_dir参数不存在", self.name)

        # 转换为Path对象
        file_dir_obj = Path(file_dir)

        # 验证导出文件目录存在
        if not file_dir_obj.exists():
            raise FileProcessingError(f"file_dir目录不存在: {file_dir}", self.name)

        return import_file_obj, file_dir_obj


    """
        执行MinerU转换(把PDF转换为MD)
    """
    def _execute_mineru(self, import_file_obj:Path, file_dir_obj:Path) -> int:
        proc = subprocess.Popen(
            args=["mineru", "-p",
                  str(import_file_obj),
                  "-o", str(file_dir_obj),
                  "--backend", "pipeline"],
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.STDOUT,  # 合并错误到标准输出
            text=True,
            encoding="utf-8",
            errors="replace",  # 遇到乱码时替换
            bufsize=1  # 行缓冲，实时输出
        )

        for line in proc.stdout:
            print(line.rstrip())

        return_code = proc.wait()
        return return_code

    """
        获取MD文件路径
    """
    def _get_md_path(self, import_file_obj:Path, file_dir_obj: Path) -> Path:
        if file_dir_obj.exists():
            return file_dir_obj / import_file_obj.stem / "auto" / (import_file_obj.stem + ".md")
        else:
            raise FileProcessingError(f"md文件不存在: {file_dir_obj / f'{import_file_obj.stem}.md'}", self.name)

if __name__ == '__main__':
    setup_logging()

    input_state = {
        "import_file_path": r"E:\文件存储\笔记\项目-掌柜智库\11_尚硅谷AI全能开发技术之项目【掌柜智库】\2.资料\pdf文档\doc\万用表RS-12的使用.pdf",
        "file_dir": r"E:\文件存储\笔记\项目-掌柜智库\11_尚硅谷AI全能开发技术之项目【掌柜智库】\2.资料\pdf文档\output",
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False
    }

    pdf_to_md_node = PdfToMdNode()

    pdf_to_md_node(input_state)
