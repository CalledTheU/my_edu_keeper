# 教育知识库问答系统

面向课程、讲义、试题和教学资料的智能知识库系统，支持文档导入、内容解析、向量检索、联网补充和流式问答。

## 功能

- PDF 转 Markdown，并保留文档结构
- Markdown 图片识别、对象存储和内容替换
- 按标题和长度切分知识片段
- 大模型识别课程或项目名称
- BGE-M3 稠密向量 + 稀疏向量混合检索
- Milvus 存储和查询，HyDE 查询扩展与 RRF 融合
- BGE Reranker 重排序
- 百炼 WebSearch MCP 联网搜索
- DeepSeek 生成答案并通过 SSE 流式返回
- MongoDB 保存会话历史，MinIO 保存文件和图片
- 导入进度和问答节点进度实时展示

## 技术栈

| 模块 | 技术 |
| --- | --- |
| Web 服务 | FastAPI、Uvicorn |
| 流程编排 | LangGraph |
| 文档解析 | MinerU |
| 图片理解 | DashScope 视觉模型 |
| 向量与重排序 | BGE-M3、BGE Reranker Large |
| 数据服务 | Milvus、MongoDB、MinIO |
| 大模型 | DeepSeek |
| 联网搜索 | 百炼 WebSearch MCP |

## 目录

```text
├─ main.py                         服务入口
├─ api/                            HTTP 路由
├─ core/                           路径和依赖配置
├─ schema/                         请求与响应模型
├─ services/                       文件导入和任务服务
├─ prompt/                         导入与问答提示词
├─ processor/import_process/        文档导入流程
├─ processor/query_process/         问答流程
├─ utils/                          向量、存储、任务和 SSE 工具
└─ front/                          导入页和问答页
```

## 运行环境

项目可使用 Python 3.10 及以上版本，建议在 IDE 中选择项目对应的虚拟环境。

```text
<项目虚拟环境>\Scripts\python.exe
```

启动前确认以下服务可访问：

- Milvus：`192.168.10.129:19530`
- MinIO：`192.168.10.129:9000`
- MongoDB：`192.168.10.129:27017`

## 配置

编辑项目根目录的 `.env`，至少配置：

```env
DEEPSEEK_API_KEY=你的DeepSeek密钥
DEEPSEEK_API_BASE=https://api.deepseek.com
DASHSCOPE_API_KEY=你的百炼通用API密钥
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
MCP_DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp

MILVUS_URL=http://192.168.10.129:19530
MONGO_URL=mongodb://用户名:密码@192.168.10.129:27017/?authSource=admin
MONGO_DB_NAME=kb_edu
MINIO_ENDPOINT=192.168.10.129:9000
MINIO_ACCESS_KEY=你的MinIO用户名
MINIO_SECRET_KEY=你的MinIO密码
MINIO_BUCKET_NAME=edu-knowledge-files
```

百炼 MCP 需要使用已开通 WebSearch 的同一账号下的有效通用 API Key。Streamable HTTP 端点对应工具名 `bailian_web_search`。

## 启动

在项目根目录执行：

```powershell
python main.py
```

启动后访问：

- 导入页面：http://127.0.0.1:8000/import
- 问答页面：http://127.0.0.1:8000/query
- 接口文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/health

## 使用流程

1. 在导入页面上传 PDF 或 Markdown 文件。
2. 等待解析、图片处理、切片、向量生成和入库完成。
3. 在问答页面提交问题。
4. 系统执行本地检索、HyDE 检索和可选联网搜索。
5. 经过 RRF、Reranker 和大模型生成最终答案。

联网搜索是可选增强能力。MCP 不可用时，系统会记录警告并继续使用本地知识库回答。

## 日志定位

```text
item_name_confirm → search_embedding / search_embedding_hyde / web_search_mcp
                 → rrf → rerank → answer_output
```

如果日志停在某个节点的“开始”之后，应查看该节点的异常堆栈。`401/403` 通常表示 Key 或权限问题，`404` 通常表示端点或协议问题，`500` 通常需要检查 MCP 账号、额度和服务状态。

## 安全提示

- `.env` 不要提交到代码仓库。
- API Key 不要粘贴到聊天、截图或日志中。
- 生产环境为数据库和 MinIO 配置独立账号及访问策略。
