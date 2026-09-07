import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile

from core.paths import get_local_base_dir
from processor.import_process.config import get_config
from processor.import_process.exceptions import FileProcessingError, MinioError
from processor.import_process.main_graph import run_import_graph, compiled_graph
from utils.client.storage_clients import StorageClients
from utils.task_util import add_running_task, add_done_task, update_task_status, TASK_STATUS_PROCESSING, \
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED

logger = logging.getLogger(__name__)

class ImportFileService:
    """
    处理上传文件业务层类
    """
    def upload_file(self, file:UploadFile):
        """
        文件上传处理: 双写: 保存到本地 & 保存到MinIO
        """

        if not file or not file.filename:
            raise FileProcessingError("上传文件不能为空")

        # 1.生成任务ID
        task_id = self.generate_task_id()
        file_name = Path(file.filename).name
        if not file_name:
            raise FileProcessingError("上传文件名无效")

        # 2.保存文件到本地
        date_path = self._get_date_path(get_local_base_dir)
        file_dir = os.path.join(date_path, task_id)
        # 创建文件夹
        os.makedirs(file_dir, exist_ok=True)

        add_running_task(task_id, node_name="upload_file")# 名称与task_util.py中 英文转中文名称一致。

        # 3.双写
        # 3.1写入本地
        local_path = self._upload_file_to_local(file, file_dir, file_name)
        
        # 3.2写入MinIO  filename用于创建MinIO文件名
        self._upload_file_to_minio(local_path, file_name)

        add_done_task(task_id, node_name="upload_file")

        return task_id, file_dir, local_path

    def run_import_graph(self,task_id, file_dir, import_file_path):
        input_state = {
            "task_id": task_id,
            "import_file_path": import_file_path,
            "file_dir": file_dir
        }
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        try:
            for event in compiled_graph.stream(input_state):
                for node_name, state in event.items():
                    print(f"任务ID: {task_id}, 运行节点: {node_name}")
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.info(f"导入流程出现异常: {e}")
            update_task_status(task_id, TASK_STATUS_FAILED, str(e))

    def generate_task_id(self):
        return uuid.uuid4().hex[:16]

    def _get_date_path(self, get_local_base_dir):
        # %Y%m%d:年月日
        # %Y:四位 %y:两位
        return os.path.join(get_local_base_dir(), datetime.now().strftime("%Y%m%d"))

    def _upload_file_to_local(self, file, file_dir, file_name=None) -> Path:
        """写入本地目录"""
        # 1. 创建文件的归属目录
        os.makedirs(file_dir, exist_ok=True)

        try:
            file_path = os.path.join(file_dir, file_name or Path(file.filename).name)
            with (open(file_path, "wb") as local_file):
                shutil.copyfileobj(file.file, local_file)
        except Exception as e:
            logger.error(f"文件保存本地出错: {e}")
            raise FileProcessingError(f"文件保存本地出错: {e}")

        return file_path

    def _upload_file_to_minio(self, file_path, file_name):
        """将上传文件保存到minio"""
        # 1.获取minio客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.warning(f"获取minio客户端出错: {e}")
            return  # 降级处理： 获取客户端失败，不影响后续流程。
        try:
            config = get_config()
            obj_name = f"origin_files/{datetime.now().strftime('%Y%m%d')}/{file_name}"
            # url = http://192.168.10.10:9000/knowledge-base-files/origin_files/20230405/万用表RS-12的使用.pdf
            minio_client.fput_object(config.minio_bucket, obj_name, file_path)
        except Exception as e:
            logger.warning(f"文件上传Minio出错: {e}")
            return  # 降级处理： 上传Minio失败，不影响后续流程。
