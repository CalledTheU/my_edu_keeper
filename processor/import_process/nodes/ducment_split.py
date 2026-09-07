import os
import re
from typing import List, Dict, Any
import json

from processor.import_process import config
from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.exceptions import StateFieldError, ValidationError
from processor.import_process.state import ImportGraphState
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    name: str = "document_split_node"  # 节点名称
    """
        文档分割节点 
    """

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """
        文档分割处理逻辑
        """

        # 1. 获取输入进行数据校验  -> 文档地址:D:\pythoncharm\pythonProject\atguigu-shopkeeper\knowledge\test\output\万用表RS-12的使用_new.md
        md_content, file_title, max_content_length, min_content_length = self._get_input_validation(state)

        # 2. 标题切分
        sections: List[Dict[str, Any]] = self._split_by_title(md_content, file_title)
        print(sections)

        # 3.切分合并  ->  长切短合
        final_sections: List[Dict[str, Any]] = self.split_and_merge(sections, max_content_length, min_content_length)

        # 4.组装切片
        chunks = self._assemble_chunk(final_sections)
        for chunk in chunks:
            chunk["course_name"] = state.get("course_name", "")
            chunk["project_name"] = state.get("project_name", "")
            chunk["chapter_name"] = chunk.get("title", "")
            chunk["source_file"] = state.get("source_file", state.get("file_title", ""))
            chunk["content_type"] = state.get("content_type", "doc_chunk")

        # 5.日志备份

        # 5.1. 日志统计
        self._log_summary(md_content, chunks, max_content_length)

        # 5.2. 备份
        self._backup_chunks(state, chunks)

        state['chunks'] = chunks

        return state

    def _get_input_validation(self, state: ImportGraphState) -> tuple[str, str, int, int]:
        md_content = state.get("md_content")  # 文档内容
        file_title = state.get("file_title")  # 文件标题
        max_content_length = self.config.max_content_length  # 切片最大长度
        min_content_length = self.config.min_content_length  # 合并短内容的最小长度

        if not md_content:
            raise StateFieldError(self.name, "md_content", str, "md_content不能为空")

        if not file_title:
            raise ValidationError("file_title不能为空", self.name)

        if max_content_length <= 0 or min_content_length <= 0 or max_content_length < min_content_length:
            raise ValidationError("最大切分和最小切分长度配置错误", self.name)

        return md_content, file_title, max_content_length, min_content_length

    def _split_by_title(self, md_content, file_title: str) -> List[Dict[str, Any]]:
        """根据标题切分文档
        md_content: 整个文档内容
        file_title: 文件标题 不带扩展名
        return [
            {
                "parent_title": "# HAK 180",
                "title": "## HAK 180 烫金机",
                "body": "产品安全手册（简体中文）...",
                "file_title": "hak180产品安全手册"
            }
        ]
        """

        sections: List[Dict[str, Any]] = []
        in_fence = False  # 是否在围栏内
        heading_re = re.compile(
            r"^\s*(#{1,6})\s+.+")  # 匹配标题正则表达式      (#{1,6})  匹配组     取标题级别：match.group(1)     原文match.group(0)
        body: List[str] = []  # 收集正文
        content_lines = md_content.split("\n")
        current_title = ""  # 当前标题
        current_level = 0  # 当前标题级别
        hierarchy = [""] * 7  # 标题层级     ["","一级","二级","三级","四级","五级"，""]

        def _flush():
            if current_title or body:
                parent_title = ""
                for lev in range(current_level - 1, 0, -1):
                    if hierarchy[lev]:
                        parent_title = hierarchy[lev]
                        break

                if not parent_title:
                    parent_title = current_title if current_title else file_title

                sections.append({
                    "parent_title": parent_title,
                    "title": current_title if current_title else file_title,  # 特殊情况，例如：处理第一个标题前的段落。这个段落没有标题，也没有父标题
                    "body": "\n".join(body),
                    "file_title": file_title
                })

        for index, line in enumerate(content_lines):
            if line.startswith("~~~") or line.startswith("```"):
                in_fence = not in_fence

            # 匹配标题
            match = heading_re.match(line) if not in_fence else None

            # 是标题
            if match:
                _flush()  # 帮我把当前标题，之前的内容封装成secion ->  List

                level = len(match.group(1))  # 标题级别   1-6 值
                current_level = level
                current_title = line
                hierarchy[current_level] = current_title

                # hierarchy = [""] * 7  # 重置，因为不重置，这里存放的都是上个段落对应的标题。 全部置空是错误的。
                for i in range(level + 1, 7):  # 只清理大于当前级别的子级别，因为这些子级别是上个段落遗留的。当前级别父级别不能清理，因为多个子对应同一个父时,其他子还要用到这个父。
                    hierarchy[i] = ""
                body = []  # 置空，只装填当前标题的正文行

            else:
                body.append(line.strip())  # 收集正文

        # 处理最后一个段落
        _flush()

        return sections

    def split_and_merge(self, sections:List[Dict[str, Any]], max_content_length:int, min_content_length:int) -> List[Dict[str, Any]]:
        """
            二次切分和合并

            Args:
                sections: 根据一级标题切分后的所有 section（章节）块
                max_content_length: 每一个 section 的 content 内容最大长度
                min_content_length: 触发合并的最小长度
        """
        self.log_step("step3", "切分及合并...")

        # 1. 切分
        current_sections = []
        for section in sections:
            #  current_sections.append([{},{}])            [[{},{}],[{},{}],[{},{}]]
            #  current_sections.extend([])                 [{},{},{},{}]
            current_sections.extend(self.split_long_section(section, max_content_length))

        # 2. 合并
        final_sections = self.merge_short_section(current_sections, min_content_length, max_content_length)
        print(final_sections)

        # 3. 返回
        return final_sections

    def split_long_section(self, section, max_content_length) -> list[Dict[str, Any]]:
        """切分长段落"""
        # 1.获取section属性值
        parent_title = section.get("parent_title")# 父标题
        title = section.get("title")# 标题
        body = section.get("body")# 正文
        file_title = section.get("file_title")# 文件标题
        sub_sections = []

        # 2.表格处理
        if "<table>" in body:
            self.logger.info("检测到表格，进行特殊处理")
            # 进行降维处理(拍扁成自然语言)
            body = MarkdownTableLinearizer.process(body)
            section["body"] = body

        # 最大标题长度限制
        MAX_TITLE_LENGTH = 50
        if len(title) > MAX_TITLE_LENGTH:
            title = title[:MAX_TITLE_LENGTH]

        # 3.定义标题前缀
        title_prefix = title + "\n\n"

        # 4.标题+正文  小于 阈值  ， 不用切分直接返回,但是也存储列表返回。
        if len(title_prefix) + len(body) <= max_content_length:
            return [section]

        # 5.获取body可用长度
        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section]

        # 6.需要切分
        if len(title_prefix) + len(body) > max_content_length:
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "。", "！", "？", ".", ",", "!", " ", ""],  # 用于切分的分隔符
                chunk_size=body_length,  # 切片最大长度
                chunk_overlap=0,  # 不重叠
                keep_separator=False,  # 不保留分隔符
            )
            texts = splitter.split_text(body)

            if len(texts) <= 1:
                return [section]

            for index,text in enumerate(texts):
                sub_sections.append({
                    "parent_title": parent_title,
                    "title": title + f"-{index+1}",
                    "body": text,
                    "file_title": file_title,
                    "part": index + 1
                })
            return sub_sections
                
    def merge_short_section(
            self,
            current_sections: list[Dict[str, Any]],
            min_content_length: int,
            max_content_length: int | None = None,
    ):
        """合并短章节内容"""
        if not current_sections:
            return []
        if max_content_length is None:
            max_content_length = self.config.max_content_length

        # 创建返回列表
        final_sections: List[Dict[str, Any]] = []

        # 获取第一个章节
        first_section = current_sections[0]
        # 开始从第二个章节遍历
        for current_section in current_sections[1:]:
            # 判断当前章节和上一个章节是否是同一个父章节，并且当前章节的正文长度是否小于阈值
            same_parent_title:bool = first_section.get("parent_title") == current_section.get("parent_title")
            merged_title = first_section.get("title", first_section.get("parent_title", ""))
            merged_body = (
                first_section.get("body", "").rstrip()
                + "\n\n"
                + current_section.get("title", "").strip()
                + "\n\n"
                + current_section.get("body", "").lstrip()
            ).strip()
            fits_max_length = len(merged_title) + 2 + len(merged_body) <= max_content_length
            if same_parent_title and len(current_section.get('body', '')) < min_content_length and fits_max_length:
                # 合并章节
                first_section['body'] = merged_body
                first_section['title'] = merged_title
                first_section.pop('part', None)
            else:
                # 不是同一个父章节，或者当前章节的正文长度大于阈值，那么就将当前章节添加到返回列表中
                final_sections.append(first_section)
                first_section = current_section
        # 添加孤儿章节(最后一个章节不会进入循环，所以需要单独添加)
        final_sections.append(first_section)

        # 对所有 section 的 part 做处理
        part_counter = {}
        # result = []
        for final_section in final_sections:
            if "part" in final_section:
                parent_title = final_section.get('parent_title')
                part_counter[parent_title] = part_counter.get(parent_title, 0) + 1
                new_part = part_counter[parent_title]
                final_section['part'] = new_part
                base_title = re.sub(r"-\d+$", "", final_section.get('title', ''))
                final_section['title'] = f"{base_title}-{new_part}"

            # result.append(final_section)

        return final_sections

    def _assemble_chunk(self, final_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """最终组合 chunk"""
        self.log_step("step4", "组装最终的切片信息...")
        chunks = []

        for chunk in final_chunks:
            # 1. 获取 chunk 的信息
            title = chunk.get('title')
            file_title = chunk.get('file_title')
            parent_title = chunk.get('parent_title')
            body = chunk.get('body')
            content = f"{title}\n\n{body}"

            # 2. 构建最终 chunk 对象
            assemble_chunk = {
                "title": title,
                "file_title": file_title,
                "parent_title": parent_title,
                "content": content,
            }

            # 3. 判断 part 是否存在
            if "part" in chunk:
                assemble_chunk['part'] = chunk.get('part')

            chunks.append(assemble_chunk)

        return chunks

    def _log_summary(self, raw_content: str, chunks: List[dict], max_length: int):
        """输出切分统计信息"""
        self.log_step("step5", "输出统计")

        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(chunks)}")
        self.logger.info(f"最大切片长度: {max_length}")

        if chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(chunks[:5]):
                title = sec.get("title", "")[:30]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(chunks) > 5:
                self.logger.info(f"  ... 还有 {len(chunks) - 5} 个章节")

    def _backup_chunks(self, state: ImportGraphState, sections: List[dict]):
        """将切分结果备份到 JSON 文件"""
        self.log_step("step6", "备份切片")

        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("未设置 file_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")

if __name__ == '__main__':
    setup_logging()

    document_node = DocumentSplitNode()
    # 构造状态字典
    file_path = r"D:\pythoncharm\pythonProject\atguigu-shopkeeper\knowledge\test\output\万用表RS-12的使用_new.md"
    # file_path = r"D:\pythoncharm\pythonProject\shopkeeper_brain260706\output\5d8cd12b-c657-4e47-aa4f-d69b281d1f74\hak180产品安全手册\auto\hak180产品安全手册_new.md"

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    state = {
        "file_title": "万用表RS-12的使用",
        "md_content": content,
        "file_dir": r"D:\pythoncharm\pythonProject\atguigu-shopkeeper\knowledge\test\output"
    }
    document_node(state)
