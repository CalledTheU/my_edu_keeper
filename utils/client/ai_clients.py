import os
import threading
from pathlib import Path
from typing import Optional

# Windows CPU 推理时避免 PyTorch/MKL 多线程触发原生访问冲突。
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from FlagEmbedding import FlagReranker

from utils.client.base import BaseClientManager, logger


load_dotenv(Path(__file__).parents[2] / ".env")


class AIClients(BaseClientManager):
    """DeepSeek 文本模型、DashScope 视觉模型和本地 BGE 模型客户端。"""

    _vision_client: Optional[OpenAI] = None
    _vision_lock = threading.Lock()

    @classmethod
    def get_openai(cls) -> OpenAI:
        """兼容旧调用：返回 DashScope 视觉客户端。"""
        return cls.get_dashscope_vision()

    @classmethod
    def get_dashscope_vision(cls) -> OpenAI:
        """获取用于图片识别的 DashScope OpenAI 兼容客户端。"""
        return cls._get_or_create("_vision_client", cls._vision_lock, cls._create_dashscope_vision)

    @classmethod
    def _create_dashscope_vision(cls) -> OpenAI:
        try:
            api_key = cls._require_env("DASHSCOPE_API_KEY")
            base_url = cls._require_env("DASHSCOPE_API_BASE")
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            logger.info("DashScope 视觉客户端初始化成功")
            return client
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"DashScope 视觉客户端初始化失败:{e}")
            raise ConnectionError(f"DashScope 视觉客户端连接失败:{e}") from e

    _deepseek_llm_text_client: Optional[ChatOpenAI] = None
    _deepseek_llm_text_lock = threading.Lock()

    _deepseek_llm_json_client: Optional[ChatOpenAI] = None
    _deepseek_llm_json_lock = threading.Lock()

    @classmethod
    def get_llm_openai(cls, response_format: bool = True) -> ChatOpenAI:
        """兼容旧调用：返回 DeepSeek 主模型客户端。"""
        return cls.get_deepseek_llm(response_format)

    @classmethod
    def get_deepseek_llm(cls, response_format: bool = True) -> ChatOpenAI:
        """获取用于对话、推理和工具调用的 DeepSeek 客户端。"""
        if response_format:
            return cls._get_or_create(
                "_deepseek_llm_json_client",
                cls._deepseek_llm_json_lock,
                lambda: cls._create_deepseek_llm(response_format),
            )
        return cls._get_or_create(
            "_deepseek_llm_text_client",
            cls._deepseek_llm_text_lock,
            lambda: cls._create_deepseek_llm(response_format),
        )

    @classmethod
    def _create_deepseek_llm(cls, response_format: bool) -> ChatOpenAI:
        try:
            api_key = cls._require_env("DEEPSEEK_API_KEY")
            base_url = cls._require_env("DEEPSEEK_API_BASE")
            model_name = cls._require_env("LLM_DEFAULT_MODEL")

            model_kwargs = {}
            if response_format:
                model_kwargs['response_format'] = {"type": "json_object"}

            client = ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=0,
                model_kwargs=model_kwargs
            )
            logger.info("DeepSeek 主模型客户端初始化成功")
            return client
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"DeepSeek 主模型客户端初始化失败:{e}")
            raise ConnectionError(f"DeepSeek 主模型连接失败:{e}") from e

    """
    BGE-M3客户端：
    """
    _bge_m3_client: Optional[BGEM3EmbeddingFunction] = None
    _bge_m3_lock = threading.Lock()

    @classmethod
    def get_bge_m3_client(cls) -> BGEM3EmbeddingFunction:
        return cls._get_or_create("_bge_m3_client", cls._bge_m3_lock, cls._create_bge_m3_client)

    @classmethod
    def _create_bge_m3_client(cls) -> BGEM3EmbeddingFunction:
        try:
            model_name = cls._require_env("BGE_M3_PATH")
            device = cls._require_env("BGE_DEVICE")
            fp16 = cls._require_env("BGE_FP16")

            bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=model_name,
                device=device,
                use_fp16=fp16,
            )
            logger.info(f"bge_m3客户端初始化成功")
            return bge_m3_ef
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"bge_m3客户端初始化失败:{e}")
            raise ConnectionError(f"bge_m3客户端创建失败:{e}") from e


    """
    BGE-M3重排序模型客户端：
    """
    _bge_m3_rerank_client: Optional[FlagReranker] = None
    _bge_m3_rerank_lock = threading.Lock()

    @classmethod
    def get_bge_m3_rerank_client(cls) -> FlagReranker:
        return cls._get_or_create("_bge_m3_rerank_client", cls._bge_m3_rerank_lock, cls._create_bge_m3_rerank_client)

    @classmethod
    def _create_bge_m3_rerank_client(cls) -> FlagReranker:
        try:
            model_name_or_path = cls._require_env("BGE_RERANKER_LARGE")
            device = os.getenv("BGE_RERANKER_DEVICE", "cpu")
            fp16_str = os.getenv("BGE_RERANKER_FP16", "0")
            fp16 = fp16_str.lower() in ("true","1")

            # CPU 不支持半精度 Reranker 推理，强制关闭以避免底层崩溃。
            if device.lower() == "cpu":
                fp16 = False

            reranker = FlagReranker( # 交叉编码器
                model_name_or_path=model_name_or_path,
                #model_name_or_path="D:\\ai_models\\modelscope_cache\\models\\BAAI\\BAAI\\bge-reranker-large",
                device=device,  # GPU 加速
                use_fp16=fp16  # 半精度推理
            )
            logger.info(f"bge_m3_rerank客户端初始化成功")
            return reranker
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"bge_m3_rerank客户端初始化失败:{e}")
            raise ConnectionError(f"bge_m3_rerank客户端创建失败:{e}") from e


if __name__ == "__main__":
    print(AIClients.get_bge_m3_rerank_client())
