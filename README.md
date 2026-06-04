# AwardAudit Agent

> 中文 | [English](#english)

一个面向高校奖项名单的审核分析 Agent。项目支持上传 CSV/XLSX 奖项表格，用户可以用中文提问，系统会通过 LLM 意图识别、LangChain tools 和 pandas 确定性计算，回答字段识别、质检审核、金额统计、排行查询和图表分析问题。

## 项目定位

这个项目不是通用表格聊天机器人，而是一个更垂直的 **高校奖项名单审核分析 Agent**。

它重点解决这些问题：

- 字段名不固定：如“住宿书院 / 宿舍书院 / 所在书院”都可能表示书院字段
- 审核问题繁琐：缺失姓名、缺失书院、重复学号、异常金额需要自动检查
- 查询问题多样：某人拿了多少钱、哪个书院金额最高、获奖最多的学生是谁
- 通用 LLM 不稳定：LLM 可以理解意图，但真实计算必须由 pandas 完成

## 核心亮点

- **LLM 意图路由**：优先让 LLM 判断应该调用哪些 tools；无 API Key 时自动使用本地 fallback
- **LangChain tools 管理**：所有分析能力都注册为 `StructuredTool`
- **灵活字段识别**：LLM 候选识别 + 程序校验 + fallback 规则，避免死匹配字段名
- **确定性业务计算**：金额求和、人数统计、排行、计数、重复检查和异常值检查都由 pandas 完成
- **轻量 pandas 查询**：开放式排行/统计问题直接在当前 DataFrame 上确定性执行
- **无 SQL 模块**：单文件场景不引入数据库，链路更短、更容易讲清楚
- **可解释结果**：返回答案、图表、字段识别、审核结果和 `tool_trace`
- **无数据库持久化**：当前版本使用内存 session，适合本地演示和简历项目

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 前端 | Vue 3, Vite |
| 后端 | FastAPI, pandas, openpyxl |
| Agent / Tool | LangChain `StructuredTool` |
| LLM 接入 | OpenAI-compatible Chat Completions API |
| 图表 | 后端生成 SVG data URL |
| 测试 | pytest, node:test |

## 架构流程

```text
上传 CSV/XLSX
    -> pandas 读取和清洗表头
    -> 生成数据 profile 和字段识别结果
    -> LLM 意图识别选择 LangChain tools
    -> pandas tools 执行确定性计算
    -> LLM 或 fallback 生成中文回答
    -> 前端展示答案、图表、审核结果和 tool_trace
```

## Tools 列表

| Tool | 作用 |
| --- | --- |
| `profile_data` | 查看行数、列数、字段类型和样例数据 |
| `analyze_data` | 分析缺失值、数值列统计和分类列分布 |
| `build_charts` | 生成最多 3 张有业务含义的图表 |
| `compare_groups` | 按分类列做分组汇总 |
| `find_outliers` | 使用 IQR 方法检测异常值 |
| `infer_award_fields` | 识别学号、姓名、书院、学院、奖项等级、奖励金额等字段 |
| `quality_check_awards` | 检查缺失字段、重复学号、异常金额 |
| `award_summary` | 汇总总金额、人均金额、书院/学院/等级排行 |
| `audit_report` | 生成结构化审核报告 |
| `query_person_awards` | 按姓名查询个人获奖明细和总金额 |
| `query_student_awards_by_id` | 按学号查询个人获奖明细和总金额 |
| `query_college_awards` | 查询某个书院的获奖情况 |
| `query_department_awards` | 查询某个学院的获奖情况 |
| `rank_students_by_amount` | 按学生汇总奖励金额排行 |
| `search_award_records` | 按关键词搜索赛事名称、奖项等级、举办单位等字段 |
| `answer_award_question` | 用 pandas 回答开放式统计、排行、次数和常见筛选问题 |

## 轻量 pandas 查询设计

项目当前只分析单个 CSV/XLSX，上传后数据已经在一个 `DataFrame` 中，因此不再引入 SQL/SQLite。

设计边界：

- LLM 只负责选择工具和辅助字段理解
- pandas 负责所有真实计算
- 常见问题通过固定函数执行：排行、计数、个人金额、书院金额、条件筛选
- 不让模型生成可执行代码，也不让模型直接计算金额
- 结果中保留 `tool_trace`，便于解释每个问题调用了哪个工具

示例：

```text
问题：拿钱第二多的是谁
路由：answer_award_question
执行：按学生分组 -> 奖励金额求和 -> 降序排序 -> 取第 2 名
回答：李四的奖励总额第 2 高，为 700.0 元。
```

## 奖项等级计数规则

奖项名单里经常会同时出现“获奖等级”和“审批依据”等字段。项目不会只按固定列名死匹配，而会先根据字段名和列值分布选择更像“审核认定等级”的列，再用 pandas 做确定性筛选。

- `国A级有几条`：精确统计国 A，不自动扩大成“国 A 及以上”
- `国B级及以上有多少`：阈值统计，包含国 A、国 B，不包含国 C、省 A
- `省B级及以上有多少`：阈值统计，包含国家级可识别记录、省 A、省 B，不包含省 C、校级
- `申请 / 审批 / 公示`：如果表格有明确状态列，会按状态过滤；如果没有状态列，默认理解为“当前名单内记录”，并在回答里说明

这样做的目标是：LLM 负责理解用户问题，程序负责保证统计口径稳定。

## 隐私与本地数据

真实奖项名单可能包含姓名、学号、学院、金额等敏感信息，请只放在本地使用。

本仓库默认忽略以下本地隐私文件：

- `backend/sample_data/*获奖*名单*.xlsx`
- `backend/sample_data/private/`
- `.env` 和 `.env.*`

可以提交到 GitHub 的只应该是脱敏示例数据，例如 `backend/sample_data/sales.csv` 和 `backend/sample_data/upload_smoke.xlsx`。提交前建议运行：

```powershell
git status --short -- backend/sample_data
git ls-files backend/sample_data
```

## 快速开始

### 1. 安装后端依赖

```powershell
python -m pip install -r backend\requirements.txt
```

### 2. 配置环境变量

复制示例环境变量：

```powershell
Copy-Item .env.example .env
```

可选配置：

```env
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek4flash
```

不配置 `DEEPSEEK_API_KEY` 也可以运行，本地 fallback 会保证 demo 可用。

项目仍兼容旧的 OpenAI-compatible 环境变量：`LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`。

### 3. 启动后端

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8002 --app-dir backend
```

也可以直接运行：

```powershell
python backend\app\main.py
```

健康检查：

```powershell
Invoke-WebRequest http://127.0.0.1:8002/health
```

### 4. 启动前端

```powershell
npm install
npm run dev
```

访问：

```text
http://localhost:5173
```

## 示例问题

上传高校奖项名单后，可以尝试：

- 这个表有哪些字段？
- 帮我识别奖项字段
- 拿钱第二多的是谁？
- 获奖最多的学生是谁？
- 哪个书院奖励总额最高？
- 国A级有几条？
- 看看国B级以上的申请有多少？
- 帮我看看有几个国A级及以上的奖被审批了
- 找一下数学建模相关奖项
- 请生成图表展示关键指标
- 生成审核报告
- 有哪些缺失值、重复学号或异常金额？

## 测试

```powershell
python -m pytest -q
npm test -- --run
npm run build
```

## 适合写进简历的表述

> 设计并实现高校奖项名单审核分析 Agent，支持 CSV/XLSX 上传、LLM 意图路由、LangChain tool 调度、字段智能识别和 pandas 确定性审核计算。系统可自动完成字段映射、缺失/重复/异常金额检查、个人/书院/学院维度统计、排行查询和审核报告生成，并通过固定 pandas 工具链保证结果可解释、可复现、易演示。

## 当前边界

- 只支持单文件上传
- session 保存在后端内存中
- 不做用户登录和权限系统
- 不持久化数据库
- 不支持多文件批量审核
- 开放式复杂问题仍受工具覆盖范围限制，但高频排行、计数和金额统计已由 pandas 工具覆盖

---

## English

AwardAudit Agent is an auditing and analytics agent for university award spreadsheets. It supports CSV/XLSX upload, Chinese natural-language questions, LLM-based intent routing, LangChain tools, and deterministic pandas computation over the uploaded file.

## What This Project Is

This is not a generic spreadsheet chatbot. It is a vertical **university award list auditing agent**.

It focuses on:

- Flexible field names, such as different variants of college/dormitory-college columns
- Award list quality checks, including missing values, duplicate student IDs, and abnormal amounts
- Business questions, such as how much a student received or which college has the highest total amount
- Stable computation by pandas instead of free-form model-generated code

## Highlights

- **LLM intent routing**: the model decides which tools to call; local fallback is used when no API key is configured
- **LangChain tool management**: all capabilities are registered as `StructuredTool`
- **Flexible field inference**: LLM candidate mapping + backend validation + rule fallback
- **Deterministic calculation**: pandas handles real aggregation, ranking, and auditing logic
- **Lightweight pandas queries**: common ranking, counting, and filtering questions run directly on the current DataFrame
- **No SQL module**: the single-file scope stays simple, explainable, and easy to debug
- **Explainable output**: answer, charts, audit result, field mapping, and `tool_trace`
- **No persistent database**: uploaded data lives in memory/session for local demo simplicity

## Tech Stack

| Layer | Stack |
| --- | --- |
| Frontend | Vue 3, Vite |
| Backend | FastAPI, pandas, openpyxl |
| Agent / Tooling | LangChain `StructuredTool` |
| LLM | OpenAI-compatible Chat Completions API |
| Charts | Backend-generated SVG data URLs |
| Testing | pytest, node:test |

## Architecture

```text
Upload CSV/XLSX
    -> Parse and clean with pandas
    -> Build data profile and infer business fields
    -> LLM selects LangChain tools
    -> pandas tools execute deterministic computation
    -> LLM or local fallback formats the answer
    -> Frontend displays answer, charts, audit result, and tool_trace
```

## Tool List

| Tool | Description |
| --- | --- |
| `profile_data` | Dataset shape, columns, types, and samples |
| `analyze_data` | Missing values and basic statistics |
| `build_charts` | Generate up to 3 meaningful charts |
| `compare_groups` | Group-by aggregation |
| `find_outliers` | IQR-based outlier detection |
| `infer_award_fields` | Infer student ID, name, college, department, award level, amount, organizer |
| `quality_check_awards` | Check missing fields, duplicate IDs, and abnormal amounts |
| `award_summary` | Total amount, average amount, and ranking summaries |
| `audit_report` | Generate a structured audit report |
| `query_person_awards` | Query one student's award records and total amount |
| `query_student_awards_by_id` | Query by student ID |
| `query_college_awards` | Query one college's award summary |
| `query_department_awards` | Query one department's award summary |
| `rank_students_by_amount` | Rank students by total award amount |
| `search_award_records` | Search award records by keyword |
| `answer_award_question` | Answer open-ended ranking, counting, and common filtering questions with pandas |

## Lightweight pandas Query Design

The project analyzes one uploaded CSV/XLSX at a time. After upload, the data already lives in a pandas `DataFrame`, so the SQL/SQLite layer has been removed.

Rules:

- The LLM selects tools and helps with flexible field understanding
- pandas performs all real aggregation, ranking, counting, and filtering
- The model never executes arbitrary Python code and never calculates money directly
- `tool_trace` is returned so the execution path remains explainable

## Award Level Counting Rules

Award spreadsheets may contain both raw award-result columns and audit-basis columns. The project does not rely on one hard-coded column name. It chooses the column that looks most like an audited award-level field by checking both column names and value distributions, then uses pandas for deterministic filtering.

- `How many national A-level awards?`: exact national A count
- `How many national B-level or above awards?`: threshold count, including national A and national B, excluding national C and provincial A
- `How many provincial B-level or above awards?`: threshold count, including recognizable national records, provincial A, and provincial B
- `Applied / approved / published`: if a status column exists, it is used as a filter; otherwise the current uploaded list is treated as the record scope and the answer includes a note

In short: the LLM understands the question, while pandas enforces the calculation.

## Privacy And Local Data

Real award lists may contain names, student IDs, departments, and award amounts. Keep them local.

The repository ignores these private local files by default:

- `backend/sample_data/*获奖*名单*.xlsx`
- `backend/sample_data/private/`
- `.env` and `.env.*`

Only anonymized toy data should be pushed to GitHub, such as `backend/sample_data/sales.csv` and `backend/sample_data/upload_smoke.xlsx`.

## Quick Start

Install backend dependencies:

```powershell
python -m pip install -r backend\requirements.txt
```

Optional environment variables:

```env
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek4flash
```

Start backend:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8002 --app-dir backend
```

Start frontend:

```powershell
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## Example Questions

- What columns does this table have?
- Identify the award fields.
- How much did a specific student receive?
- Who received the second highest total amount?
- Which student has the most award records?
- Which college has the highest total award amount?
- How many exact national A-level awards are there?
- How many national B-level or above award applications are there?
- How many approved national A-level awards are there?
- Generate charts for key metrics.
- Generate an audit report.
- Check missing values, duplicate student IDs, and abnormal amounts.

## Tests

```powershell
python -m pytest -q
npm test -- --run
npm run build
```

## Current Limitations

- Single-file upload only
- In-memory session storage
- No login or permission system
- No persistent database
- No multi-file batch processing
- Open-ended complex questions are limited by the current tool coverage, while common ranking, counting, and amount questions are handled by pandas tools
