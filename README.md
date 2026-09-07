# atguigu_edu_keeper

尚硅谷实战项目 · **掌柜智库 · 教育版**

在模板项目 `atguigu-shopkeeper/knowledge`（掌柜智库，RAG 智能问答知识库）基础上改造：
**技术栈完全沿用**（FastAPI + LangGraph 编排 + MinerU 文档解析 + Qwen-VL 图片理解 +
BGE-M3 混合检索 + Milvus 向量库 + MongoDB 历史 + MinIO 对象存储 + DeepSeek 大模型），
仅把业务内容域从「掌柜/商品/电商」替换为「教育」场景。

> 当前状态：**整体框架（目录骨架 + 占位文件）已就绪**，各模块待按开发路线逐个实现。

---

## 一、与模板的差异（布局改造）

| 差异点 | 模板 atguigu-shopkeeper | 本仓库 atguigu_edu_keeper |
| --- | --- | --- |
| 代码位置 | 仓库根下套一层包 `knowledge/` | **代码平铺在仓库根**（无顶层包名） |
| 包内导入 | `from knowledge.core.paths import ...` | `from core.paths import ...`（去掉前缀） |
| 服务入口 | `python -m knowledge.api.import_router` | 根目录 `main.py`（`python main.py`） |
| 存储命名 | 库 `kb001`、集合 `kb_chunks_v1`、桶 `knowledge-base-files` | 库 `kb_edu`、集合 `edu_*_v1`、桶 `edu-knowledge-files`（已配在 `.env`） |
| 业务对象 | 商品（`item_name`）/ 掌柜客服 | 教育场景对象（课程/讲义/试题等，**命名待定，见改造点清单**） |

> ⚠️ 模板中 `processor/import_process/nodes/ducment_split.py` 的文件名拼写与 `document` 不同，
> 本骨架**保留该拼写以对齐模板**，方便逐文件对照。

---

## 二、目录结构与模板对照

```
atguigu_edu_keeper/
├─ main.py                     # FastAPI 总入口(平铺布局新增)   ← 模板无,汇总两个 router
├─ requirements.txt            # 依赖清单(与模板一致)
├─ .env / .env.example         # 环境变量(教育版独立存储命名)
│
├─ api/                        # ← 模板 knowledge/api
│  ├─ import_router.py         #   导入侧: /import /upload /status/{task_id}
│  └─ query_router.py          #   问答侧(占位,模板中亦为空): 规划 SSE 问答/历史
├─ core/                       # ← 模板 knowledge/core
│  ├─ paths.py                 #   项目根/临时目录/前端目录常量
│  └─ deps.py                  #   依赖注入(单例业务服务)
├─ schema/                     # ← 模板 knowledge/schema
│  ├─ upload_schema.py         #   UploadResponse / TaskStatusResponse(已按模板实现)
│  ├─ task_schema.py           #   任务状态(占位)
│  └─ query_schema.py          #   问答请求/响应模型(占位)
├─ services/                   # ← 模板 knowledge/services
│  ├─ file_import_service.py   #   ImportFileService: 上传(本地+MinIO 双写)/触发导入图
│  └─ task_service.py          #   任务业务(占位,模板中为空)
├─ prompt/                     # ← 模板 knowledge/prompt
│  ├─ import_prompt.py         #   导入提示词(对象名抽取/信息归纳,教育版需改写)
│  └─ query_prompt.py          #   问答提示词(HyDE/抽取/答案输出,教育版需改写)
├─ processor/                  # ← 模板 knowledge/processor
│  ├─ import_process/          #   ★ 知识导入流程(LangGraph)
│  │  ├─ base.py / config.py / exceptions.py / state.py / main_graph.py
│  │  └─ nodes/                #   entry → pdf_to_md → md_img → ducment_split
│  │                           #       → item_name_recognition → bge_embedding → import_milvus
│  └─ query_process/           #   ★ 知识问答流程(LangGraph)
│     ├─ base.py / config.py / exceptions.py / state.py / prompt.py / main_graph.py
│     └─ nodes/                #   item_name_confirm → vector_search/hyde_search
│                              #       → rrf → rerank → web_search_mcp → answer_output
├─ utils/                      # ← 模板 knowledge/utils
│  ├─ embedding_util.py  milvus_util.py  mongo_history_util.py
│  ├─ task_util.py  sse_util.py  markdown_util.py
│  └─ client/                  #   base / ai_clients / storage_clients(单例客户端)
└─ front/                      # ← 模板 knowledge/front
   └─ import.html              #   导入前端页(当前为占位页)
```

各占位文件头注释均标注了**对应模板文件路径**，实现时可逐文件对照。

---

## 三、改造点清单（教育版 TODO）

- [ ] **对象命名语义**：模板核心对象为「商品名 item_name / 掌柜客服」。教育版改为哪类对象
      （如：课程、讲义、试题、学科知识点……）——决定 `item_name_recognition`、
      `item_name_confirm`、`state.item_name`、Milvus 集合结构及 prompt 措辞，命名后需全局替换。
- [ ] **知识资料**：模板 `knowledge/interview/` 为「掌柜智库-高频面试题.docx」，
      教育版替换为对应的高频问答/知识文档。
- [ ] **prompt 改写**：`prompt/import_prompt.py`、`prompt/query_prompt.py` 中的角色设定
      （客服助手 → 教育场景角色）与抽取目标（商品名 → 教育对象名）。
- [ ] **前端页面**：`front/import.html` 按模板重写（含上传进度/SSE 实时日志）。
- [ ] **导入链路**：MinerU(PDF→MD) → 图片(VLM 描述+MinIO) → 标题分块 →
      对象名抽取 → BGE-M3 混合向量 → Milvus 入库。
- [ ] **问答链路**：对象名对齐 → 混合检索(向量+HyDE) → RRF → 重排序 →
      联网补全(MCP) → DeepSeek 生成答案(SSE) + MongoDB 历史。

---

## 四、开发路线（镜像模板 git 提交顺序）

1. `core/ schema/ utils/` 基础设施（路径、数据模型、客户端单例、任务/SSE 容器）
2. `import_process`：base/config/exceptions/state → entry 节点 → pdf_to_md(MinerU)
3. md_img（图片扫描 + VLM 摘要 + MinIO 上传替换）
4. ducment_split（按标题分块 + 长块拆分 + 短块合并）
5. item_name_recognition（对象名识别 + 向量化 + 入 Milvus）
6. bge_embedding + import_milvus（切片向量化 + 建集入库）
7. `services.file_import_service` + `api.import_router`（上传→异步跑图→查状态）
8. `query_process` 各节点 + `api.query_router`（SSE 流式问答）
9. `front/` 前端与整体联调

---

## 五、运行方式（骨架自检）

```powershell
# 1) 安装基础依赖(骨架只需 fastapi/uvicorn;完整链路按 requirements.txt 安装,含 torch/mineru 等重型包)
.\.venv\Scripts\python -m pip install fastapi uvicorn

# 2) 启动服务
.\.venv\Scripts\python main.py

# 3) 访问
#    Swagger:  http://127.0.0.1:8000/docs
#    健康检查: http://127.0.0.1:8000/health
#    导入页:   http://127.0.0.1:8000/import  (占位页)
```

> 环境变量在 `.env` 中配置（已被 git 忽略）。完整安装重型依赖建议在
> 实现对应模块时再执行，避免一次性下载 torch/mineru 全家桶。
