# AwardAudit Agent

> 中文 | [English](#english)

一个面向高校奖项名单的审核分析 Agent。项目支持上传 CSV/XLSX 奖项表格，用户可以用中文提问，系统会通过 LLM 意图识别、LangChain tools、pandas 确定性计算和安全只读 SQL，回答字段识别、质检审核、金额统计、排行查询和图表分析问题。

## 项目定位

这个项目不是通用表格聊天机器人，而是一个更垂直的 **高校奖项名单审核分析 Agent**。

它重点解决这些问题：

- 字段名不固定：如“住宿书院 / 宿舍书院 / 所在书院”都可能表示书院字段
- 审核问题繁琐：缺失姓名、缺失书院、重复学号、异常金额需要自动检查
- 查询问题多样：某人拿了多少钱、哪个书院金额最高、获奖最多的学生是谁
- 通用 LLM 不稳定：LLM 可以理解意图，但真实计算必须由 pandas / SQL 完成

## 核心亮点

- **LLM 意图路由**：优先让 LLM 判断应该调用哪些 tools；无 API Key 时自动使用本地 fallback
- **LangChain tools 管理**：所有分析能力都注册为 `StructuredTool`
- **灵活字段识别**：LLM 候选识别 + 程序校验 + fallback 规则，避免死匹配字段名
- **确定性业务计算**：金额求和、人数统计、重复检查、异常值检查都由 pandas 完成
- **安全 Text-to-SQL**：开放式排行/统计问题会进入临时 SQLite，只允许只读 SQL
- **QueryPlan SQL 模板**：高频排行问题使用结构化 QueryPlan 生成稳定 SQL
- **可解释结果**：返回答案、图表、字段识别、审核结果和 `tool_trace`
- **无数据库持久化**：当前版本使用内存 session 和临时 SQLite，适合本地演示和简历项目

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 前端 | Vue 3, Vite |
| 后端 | FastAPI, pandas, openpyxl |
| Agent / Tool | LangChain `StructuredTool` |
| LLM 接入 | OpenAI-compatible Chat Completions API |
| SQL 查询 | SQLite in-memory database |
| 图表 | 后端生成 SVG data URL |
| 测试 | pytest, node:test |

## 架构流程

```text
上传 CSV/XLSX
    -> pandas 读取和清洗表头
    -> 生成数据 profile 和字段识别结果
    -> LLM 意图识别选择 LangChain tools
    -> pandas tools / 安全 SQL tool 执行计算
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
| `answer_by_sql` | 将开放式统计问题转成安全只读 SQL 查询 |

## 安全 SQL 设计

项目支持 Text-to-SQL，但不会让模型直接无限制操作数据库。

安全边界：

- 只允许 `SELECT` / `WITH`
- 禁止 `INSERT`、`UPDATE`、`DELETE`、`DROP`、`ALTER`、`CREATE` 等危险语句
- 只允许查询临时表 `awards`
- 自动补 `LIMIT`
- 使用内存 SQLite，不持久化用户上传数据
- 高频排行问题优先走 QueryPlan + SQL 模板，而不是完全依赖 LLM 生成 SQL

示例：

```text
问题：拿钱第二多的是谁
QueryPlan:
{
  "intent": "rank_students",
  "entity": "student",
  "metric": "sum_amount",
  "rank_index": 2,
  "top_n": null
}
```

生成 SQL：

```sql
SELECT "姓名", SUM("奖励金额（元）") AS total_amount, COUNT(*) AS record_count
FROM awards
WHERE "姓名" IS NOT NULL AND TRIM(CAST("姓名" AS TEXT)) != ""
GROUP BY "姓名"
ORDER BY total_amount DESC, "姓名" ASC
LIMIT 1 OFFSET 1
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
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

不配置 `LLM_API_KEY` 也可以运行，本地 fallback 会保证 demo 可用。

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

当前验证结果：

```text
backend: 52 passed
frontend tests: passed
frontend build: passed
```

## 适合写进简历的表述

> 设计并实现高校奖项名单审核分析 Agent，支持 CSV/XLSX 上传、LLM 意图路由、LangChain tool 调度、字段智能识别、pandas 确定性审核计算和安全 Text-to-SQL 查询。系统可自动完成字段映射、缺失/重复/异常金额检查、个人/书院/学院维度统计、排行查询和审核报告生成，并通过只读 SQL 校验和 QueryPlan 模板提升查询安全性与稳定性。

## 当前边界

- 只支持单文件上传
- session 保存在后端内存中
- 不做用户登录和权限系统
- 不持久化数据库
- 不支持多文件批量审核
- 开放式 SQL 问题仍依赖 LLM 质量，但高频排行问题已模板化

---

## English

AwardAudit Agent is an auditing and analytics agent for university award spreadsheets. It supports CSV/XLSX upload, Chinese natural-language questions, LLM-based intent routing, LangChain tools, deterministic pandas computation, and safe read-only SQL over a temporary SQLite table.

## What This Project Is

This is not a generic spreadsheet chatbot. It is a vertical **university award list auditing agent**.

It focuses on:

- Flexible field names, such as different variants of college/dormitory-college columns
- Award list quality checks, including missing values, duplicate student IDs, and abnormal amounts
- Business questions, such as how much a student received or which college has the highest total amount
- Stable computation by pandas and SQL instead of free-form model-generated code

## Highlights

- **LLM intent routing**: the model decides which tools to call; local fallback is used when no API key is configured
- **LangChain tool management**: all capabilities are registered as `StructuredTool`
- **Flexible field inference**: LLM candidate mapping + backend validation + rule fallback
- **Deterministic calculation**: pandas handles real aggregation, ranking, and auditing logic
- **Safe Text-to-SQL**: open-ended ranking and statistics questions run on temporary SQLite with read-only validation
- **QueryPlan SQL templates**: common ranking queries use deterministic SQL generation
- **Explainable output**: answer, charts, audit result, field mapping, and `tool_trace`
- **No persistent database**: uploaded data lives in memory/session for local demo simplicity

## Tech Stack

| Layer | Stack |
| --- | --- |
| Frontend | Vue 3, Vite |
| Backend | FastAPI, pandas, openpyxl |
| Agent / Tooling | LangChain `StructuredTool` |
| LLM | OpenAI-compatible Chat Completions API |
| SQL | In-memory SQLite |
| Charts | Backend-generated SVG data URLs |
| Testing | pytest, node:test |

## Architecture

```text
Upload CSV/XLSX
    -> Parse and clean with pandas
    -> Build data profile and infer business fields
    -> LLM selects LangChain tools
    -> pandas tools / safe SQL tool execute computation
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
| `answer_by_sql` | Answer open-ended statistical questions with safe read-only SQL |

## Safe SQL Design

The project supports Text-to-SQL, but the model is not allowed to freely operate on a database.

Safety rules:

- Only `SELECT` / `WITH` are allowed
- `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, and similar statements are blocked
- SQL can only query the temporary `awards` table
- `LIMIT` is enforced
- SQLite is in-memory only
- Common ranking questions use QueryPlan-based SQL templates instead of fully free-form LLM SQL

## Quick Start

Install backend dependencies:

```powershell
python -m pip install -r backend\requirements.txt
```

Optional environment variables:

```env
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
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
- Open-ended SQL questions still depend on model quality, while common ranking questions are template-based
