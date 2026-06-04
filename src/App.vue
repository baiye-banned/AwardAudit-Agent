<script setup>
import { computed, ref } from 'vue'
import { askAgent, uploadDataset } from './api/analysis'

const selectedFile = ref(null)
const sessionId = ref('')
const profile = ref(null)
const question = ref('')
const messages = ref([])
const charts = ref([])
const toolTrace = ref([])
const auditResult = ref(null)
const loading = ref(false)
const errorMessage = ref('')

const fieldLabels = {
  student_id: '学号',
  name: '姓名',
  college: '书院',
  department: '学院',
  award_level: '奖项等级',
  amount: '奖励金额',
  organizer: '举办单位',
}

const canAsk = computed(() => {
  return sessionId.value && question.value.trim() && !loading.value
})

const selectedFileName = computed(() => {
  return selectedFile.value ? selectedFile.value.name : '选择 CSV / XLSX 文件'
})

const numericColumnText = computed(() => {
  return shortList(profile.value?.numeric_columns || [])
})

const textColumnText = computed(() => {
  return shortList(profile.value?.text_columns || [])
})

const tableRows = computed(() => {
  return profile.value?.all_rows || profile.value?.sample_rows || []
})

const awardFields = computed(() => {
  return profile.value?.award_fields || auditResult.value?.fields || {}
})

const summary = computed(() => {
  return auditResult.value?.summary || null
})

const qualityIssues = computed(() => {
  return auditResult.value?.quality_issues || []
})

const exampleQuestions = [
  '生成审核报告',
  '检查重复学生和缺失字段',
  '哪个书院奖励总额最高？',
  '请生成图表展示关键指标',
]

function shortList(items) {
  if (!items.length) {
    return '无'
  }

  const firstItems = items.slice(0, 3).join('、')
  const restCount = items.length - 3
  return restCount > 0 ? `${firstItems} 等 ${items.length} 列` : firstItems
}

function formatCell(value) {
  if (value === null || value === undefined || value === '') {
    return '-'
  }

  return String(value)
}

function confidenceText(value) {
  if (value === 'high') return '高'
  if (value === 'medium') return '中'
  return '低'
}

function handleFileChange(event) {
  selectedFile.value = event.target.files[0] || null
  errorMessage.value = ''
}

async function handleUpload() {
  if (!selectedFile.value) {
    errorMessage.value = '请先选择 CSV 或 XLSX 文件'
    return
  }

  loading.value = true
  errorMessage.value = ''

  try {
    const data = await uploadDataset(selectedFile.value)
    sessionId.value = data.session_id
    profile.value = data.profile
    messages.value = []
    charts.value = []
    toolTrace.value = []
    auditResult.value = null
  } catch (error) {
    errorMessage.value = error.message
  } finally {
    loading.value = false
  }
}

async function handleAsk() {
  if (!canAsk.value) {
    return
  }

  const userQuestion = question.value.trim()
  question.value = ''
  await sendQuestion(userQuestion)
}

async function sendQuestion(userQuestion) {
  errorMessage.value = ''
  loading.value = true
  messages.value.push({ role: 'user', content: userQuestion })

  try {
    const data = await askAgent({
      sessionId: sessionId.value,
      question: userQuestion,
      history: messages.value,
    })

    messages.value.push({ role: 'assistant', content: data.answer })
    charts.value = data.charts || []
    toolTrace.value = data.tool_trace || []
    profile.value = data.profile || profile.value
    auditResult.value = data.audit_result || auditResult.value
  } catch (error) {
    errorMessage.value = error.message
  } finally {
    loading.value = false
  }
}

function useExample(text) {
  question.value = text
}

function generateAuditReport() {
  if (!sessionId.value || loading.value) {
    return
  }

  sendQuestion('生成审核报告')
}
</script>

<template>
  <main class="app-shell">
    <section class="hero">
      <div class="hero-copy">
        <p class="eyebrow">Award Audit Agent</p>
        <h1>高校奖项名单审核 Agent</h1>
        <p class="hero-text">
          上传奖项名单，自动识别字段、检查异常、汇总奖励金额，并生成可复核的审核报告。
        </p>
      </div>

      <div class="upload-card">
        <label class="file-picker">
          <input type="file" accept=".csv,.xlsx,.xls" @change="handleFileChange" />
          <span class="file-icon">+</span>
          <span class="file-name" :title="selectedFileName">{{ selectedFileName }}</span>
        </label>
        <button class="primary-button" type="button" :disabled="loading" @click="handleUpload">
          {{ loading ? '处理中' : '上传数据' }}
        </button>
      </div>
    </section>

    <p v-if="errorMessage" class="error-banner">{{ errorMessage }}</p>

    <section class="metric-grid">
      <article class="metric-card">
        <span>行数</span>
        <strong>{{ profile?.row_count || '-' }}</strong>
      </article>
      <article class="metric-card">
        <span>列数</span>
        <strong>{{ profile?.column_count || '-' }}</strong>
      </article>
      <article class="metric-card">
        <span>数值列</span>
        <strong :title="profile?.numeric_columns?.join('、')">{{ numericColumnText }}</strong>
      </article>
      <article class="metric-card">
        <span>文本列</span>
        <strong :title="profile?.text_columns?.join('、')">{{ textColumnText }}</strong>
      </article>
    </section>

    <section class="audit-panel">
      <div class="panel-head">
        <div>
          <p class="panel-kicker">Audit Workspace</p>
          <h2>审核工作台</h2>
          <p class="panel-note">字段由 LLM 候选识别和程序校验共同确认，低置信度字段会直接标出。</p>
        </div>
        <button class="secondary-button" type="button" :disabled="!sessionId || loading" @click="generateAuditReport">
          生成审核报告
        </button>
      </div>

      <div class="audit-grid">
        <section class="audit-card">
          <h3>字段识别</h3>
          <div class="field-list">
            <div v-for="(label, key) in fieldLabels" :key="key" class="field-row">
              <span>{{ label }}</span>
              <strong>{{ awardFields[key]?.column || '未识别' }}</strong>
              <em :class="['confidence', awardFields[key]?.confidence || 'low']">
                {{ confidenceText(awardFields[key]?.confidence) }}
              </em>
            </div>
          </div>
        </section>

        <section class="audit-card">
          <h3>质检问题</h3>
          <p v-if="!qualityIssues.length" class="muted">生成审核报告后，这里会显示缺失、重复和异常金额。</p>
          <ul v-else class="issue-list">
            <li v-for="issue in qualityIssues.slice(0, 8)" :key="issue.type + issue.column">
              <strong>{{ issue.message }}</strong>
              <span>{{ issue.count }} 条</span>
            </li>
          </ul>
        </section>

        <section class="audit-card">
          <h3>关键指标</h3>
          <p v-if="!summary" class="muted">生成审核报告后，这里会显示总金额和排行。</p>
          <div v-else class="summary-list">
            <div><span>总奖励金额</span><strong>{{ summary.total_amount }}</strong></div>
            <div><span>人均奖励金额</span><strong>{{ summary.average_amount }}</strong></div>
            <div><span>记录数</span><strong>{{ summary.total_rows }}</strong></div>
          </div>
        </section>
      </div>

      <pre v-if="auditResult?.report" class="report-box">{{ auditResult.report }}</pre>
    </section>

    <section class="workbench">
      <section class="data-panel">
        <div class="panel-head">
          <div>
            <p class="panel-kicker">Dataset Preview</p>
            <h2>数据预览</h2>
            <p v-if="profile" class="panel-note">显示全部 {{ tableRows.length }} 行，可横向和纵向滚动查看。</p>
          </div>
          <span v-if="profile" class="status-pill">已载入</span>
        </div>

        <div v-if="!profile" class="empty-state">
          <p>上传 CSV / XLSX 后，这里会显示全部数据。</p>
        </div>

        <div v-else class="table-shell">
          <table :style="{ minWidth: `${Math.max(profile.columns.length * 190, 900)}px` }">
            <thead>
              <tr>
                <th v-for="column in profile.columns" :key="column" :title="column">
                  {{ column }}
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, index) in tableRows" :key="index">
                <td
                  v-for="column in profile.columns"
                  :key="column"
                  :title="formatCell(row[column])"
                >
                  {{ formatCell(row[column]) }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="agent-panel">
        <div class="panel-head">
          <div>
            <p class="panel-kicker">Agent Workspace</p>
            <h2>向 Agent 提问</h2>
          </div>
          <span class="status-pill">{{ sessionId ? '可提问' : '待上传' }}</span>
        </div>

        <div class="example-grid">
          <button
            v-for="item in exampleQuestions"
            :key="item"
            type="button"
            @click="useExample(item)"
          >
            {{ item }}
          </button>
        </div>

        <div v-if="toolTrace.length" class="trace-panel">
          <div class="trace-title">Agent 执行过程</div>
          <ol>
            <li v-for="item in toolTrace" :key="item.tool">
              <span>{{ item.tool }}</span>
              <em>{{ item.status }}</em>
            </li>
          </ol>
        </div>

        <div class="messages">
          <p v-if="messages.length === 0" class="muted">
            还没有对话。上传数据后，可以先点击“生成审核报告”。
          </p>
          <article
            v-for="(message, index) in messages"
            :key="index"
            :class="['message', message.role]"
          >
            <strong>{{ message.role === 'user' ? '你' : 'Agent' }}</strong>
            <pre>{{ message.content }}</pre>
          </article>
        </div>

        <form class="ask-form" @submit.prevent="handleAsk">
          <input
            v-model="question"
            type="text"
            placeholder="例如：检查重复学生，并生成审核报告"
          />
          <button type="submit" :disabled="!canAsk">
            {{ loading ? '分析中' : '发送' }}
          </button>
        </form>
      </section>
    </section>

    <section v-if="charts.length" class="chart-panel">
      <div class="panel-head">
        <div>
          <p class="panel-kicker">Auto Charts</p>
          <h2>自动生成图表</h2>
        </div>
      </div>
      <div class="chart-grid">
        <figure v-for="chart in charts" :key="chart.title">
          <img :src="chart.image_base64" :alt="chart.title" />
          <figcaption>{{ chart.title }}</figcaption>
        </figure>
      </div>
    </section>
  </main>
</template>

<style scoped>
:global(body) {
  overflow-x: hidden;
  background:
    radial-gradient(circle at top left, rgba(42, 117, 255, 0.13), transparent 32rem),
    linear-gradient(135deg, #f6f8fb 0%, #eef2f7 48%, #f8f5ee 100%);
}

button {
  border: 0;
  cursor: pointer;
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.58;
}

.app-shell {
  width: min(1440px, calc(100vw - 40px));
  min-width: 0;
  margin: 0 auto;
  padding: 28px 0 42px;
  color: #111827;
}

.hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(360px, 480px);
  gap: 18px;
  align-items: stretch;
  margin-bottom: 18px;
}

.hero-copy,
.upload-card,
.metric-card,
.data-panel,
.agent-panel,
.chart-panel,
.audit-panel {
  border: 1px solid rgba(17, 24, 39, 0.08);
  box-shadow: 0 18px 50px rgba(17, 24, 39, 0.08);
}

.hero-copy {
  min-width: 0;
  padding: 28px 30px;
  color: #f8fafc;
  background: linear-gradient(120deg, rgba(14, 23, 42, 0.96), rgba(24, 46, 80, 0.94));
  border-radius: 8px;
}

.eyebrow,
.panel-kicker {
  margin: 0;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0;
  text-transform: uppercase;
}

.eyebrow {
  color: #93c5fd;
}

h1,
h2,
h3 {
  margin: 0;
  letter-spacing: 0;
}

h1 {
  margin-top: 8px;
  font-size: 34px;
  line-height: 1.12;
}

h2 {
  font-size: 20px;
  line-height: 1.2;
}

h3 {
  font-size: 16px;
}

.hero-text {
  width: min(620px, 100%);
  margin: 12px 0 0;
  color: rgba(248, 250, 252, 0.76);
}

.upload-card {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 120px;
  gap: 12px;
  align-content: center;
  min-width: 0;
  padding: 22px;
  background: rgba(255, 255, 255, 0.86);
  border-radius: 8px;
  backdrop-filter: blur(18px);
}

.file-picker {
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr);
  gap: 10px;
  align-items: center;
  min-width: 0;
  padding: 10px;
  background: #f8fafc;
  border: 1px dashed #b6c3d8;
  border-radius: 8px;
  cursor: pointer;
}

.file-picker input {
  position: absolute;
  width: 1px;
  height: 1px;
  opacity: 0;
  pointer-events: none;
}

.file-icon {
  display: grid;
  width: 34px;
  height: 34px;
  place-items: center;
  color: #ffffff;
  background: #1f6feb;
  border-radius: 8px;
  font-size: 22px;
}

.file-name {
  min-width: 0;
  overflow: hidden;
  color: #1f2937;
  font-weight: 700;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.primary-button,
.secondary-button,
.ask-form button {
  color: #ffffff;
  background: #1f6feb;
  border-radius: 8px;
  font-weight: 800;
}

.primary-button {
  min-height: 56px;
}

.secondary-button {
  flex: 0 0 auto;
  padding: 10px 14px;
}

.error-banner {
  margin: 0 0 16px;
  padding: 12px 14px;
  color: #8f1d1d;
  background: #fff1f0;
  border: 1px solid #ffc9c4;
  border-radius: 8px;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin-bottom: 18px;
}

.metric-card {
  min-width: 0;
  min-height: 118px;
  padding: 18px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 8px;
}

.metric-card span {
  display: block;
  color: #64748b;
  font-size: 13px;
  font-weight: 700;
}

.metric-card strong {
  display: -webkit-box;
  max-height: 64px;
  margin-top: 10px;
  overflow: hidden;
  color: #0f172a;
  font-size: 24px;
  line-height: 1.32;
  text-overflow: ellipsis;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  word-break: break-word;
}

.audit-panel,
.data-panel,
.agent-panel,
.chart-panel {
  min-width: 0;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.92);
  border-radius: 8px;
}

.audit-panel {
  margin-bottom: 18px;
}

.panel-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding: 18px 20px;
  border-bottom: 1px solid #e7edf5;
}

.panel-kicker {
  margin-bottom: 6px;
  color: #1f6feb;
}

.panel-note,
.muted {
  margin: 8px 0 0;
  color: #64748b;
  font-size: 13px;
}

.audit-grid {
  display: grid;
  grid-template-columns: 1.2fr 1fr 0.9fr;
  gap: 14px;
  padding: 18px 20px;
}

.audit-card {
  min-width: 0;
  padding: 14px;
  background: #f8fafc;
  border: 1px solid #e7edf5;
  border-radius: 8px;
}

.field-list {
  display: grid;
  gap: 8px;
  margin-top: 12px;
}

.field-row {
  display: grid;
  grid-template-columns: 78px minmax(0, 1fr) 42px;
  gap: 8px;
  align-items: center;
  font-size: 13px;
}

.field-row span {
  color: #64748b;
}

.field-row strong {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.confidence {
  padding: 3px 6px;
  border-radius: 999px;
  font-size: 12px;
  font-style: normal;
  text-align: center;
}

.confidence.high {
  color: #14532d;
  background: #dcfce7;
}

.confidence.medium {
  color: #92400e;
  background: #fef3c7;
}

.confidence.low {
  color: #991b1b;
  background: #fee2e2;
}

.issue-list {
  display: grid;
  gap: 8px;
  margin: 12px 0 0;
  padding: 0;
  list-style: none;
}

.issue-list li,
.summary-list div {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  padding: 9px 10px;
  background: #ffffff;
  border: 1px solid #e7edf5;
  border-radius: 8px;
}

.issue-list span,
.summary-list span {
  flex: 0 0 auto;
  color: #64748b;
}

.summary-list {
  display: grid;
  gap: 8px;
  margin-top: 12px;
}

.report-box {
  margin: 0 20px 20px;
  padding: 14px;
  overflow: auto;
  color: #172033;
  background: #f8fafc;
  border: 1px solid #e7edf5;
  border-radius: 8px;
  white-space: pre-wrap;
  font-family: inherit;
}

.workbench {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(360px, 0.9fr);
  gap: 18px;
  align-items: start;
}

.data-panel,
.agent-panel {
  height: min(680px, calc(100vh - 260px));
  min-height: 520px;
}

.status-pill {
  flex: 0 0 auto;
  padding: 5px 9px;
  color: #14532d;
  background: #dcfce7;
  border: 1px solid #bbf7d0;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 800;
}

.empty-state {
  display: grid;
  height: calc(100% - 77px);
  place-items: center;
  padding: 24px;
  color: #64748b;
  text-align: center;
}

.table-shell {
  width: 100%;
  height: calc(100% - 77px);
  overflow-x: scroll;
  overflow-y: auto;
  overscroll-behavior: contain;
  scrollbar-gutter: stable both-edges;
}

table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
  font-size: 14px;
}

th,
td {
  width: 180px;
  min-width: 180px;
  max-width: 180px;
  padding: 11px 12px;
  overflow: hidden;
  border-bottom: 1px solid #e7edf5;
  text-align: left;
  text-overflow: ellipsis;
  white-space: nowrap;
}

th {
  position: sticky;
  top: 0;
  z-index: 1;
  color: #334155;
  background: #f8fafc;
  font-weight: 800;
}

.agent-panel {
  display: flex;
  flex-direction: column;
}

.example-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  padding: 16px 20px 0;
}

.example-grid button {
  min-width: 0;
  padding: 10px 12px;
  overflow: hidden;
  color: #1e3a8a;
  background: #edf4ff;
  border: 1px solid #d7e6ff;
  border-radius: 8px;
  font-weight: 700;
  text-align: left;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.trace-panel {
  margin: 14px 20px 0;
  padding: 12px;
  background: #f8fafc;
  border: 1px solid #e7edf5;
  border-radius: 8px;
}

.trace-title {
  color: #475569;
  font-size: 13px;
  font-weight: 800;
}

.trace-panel ol {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 10px 0 0;
  padding: 0;
  list-style: none;
}

.trace-panel li {
  display: flex;
  gap: 7px;
  max-width: 100%;
  padding: 6px 8px;
  overflow: hidden;
  background: #ffffff;
  border: 1px solid #e7edf5;
  border-radius: 7px;
  font-size: 12px;
}

.trace-panel em {
  flex: 0 0 auto;
  color: #15803d;
  font-style: normal;
  font-weight: 800;
}

.messages {
  flex: 1;
  min-height: 0;
  margin: 14px 20px 0;
  padding-right: 4px;
  overflow-y: auto;
}

.message {
  max-width: 92%;
  margin-bottom: 12px;
  padding: 12px 13px;
  border-radius: 8px;
}

.message.user {
  margin-left: auto;
  color: #ffffff;
  background: #1f6feb;
}

.message.assistant {
  color: #172033;
  background: #f6f8fb;
  border: 1px solid #e7edf5;
}

.message strong {
  display: block;
  margin-bottom: 6px;
  font-size: 13px;
}

.message pre {
  max-width: 100%;
  margin: 0;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: inherit;
}

.ask-form {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 88px;
  gap: 10px;
  padding: 16px 20px 20px;
  border-top: 1px solid #e7edf5;
}

.ask-form input {
  min-width: 0;
  padding: 12px 13px;
  background: #ffffff;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  outline: none;
}

.chart-panel {
  margin-top: 18px;
}

.chart-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  padding: 18px 20px 20px;
}

figure {
  min-width: 0;
  margin: 0;
  padding: 12px;
  background: #f8fafc;
  border: 1px solid #e7edf5;
  border-radius: 8px;
}

figcaption {
  margin-top: 10px;
  overflow: hidden;
  color: #475569;
  font-size: 13px;
  font-weight: 700;
  text-overflow: ellipsis;
  white-space: nowrap;
}

@media (max-width: 1100px) {
  .hero,
  .workbench,
  .audit-grid {
    grid-template-columns: 1fr;
  }

  .upload-card {
    grid-template-columns: minmax(0, 1fr) 120px;
  }

  .data-panel,
  .agent-panel {
    height: auto;
    min-height: 460px;
  }

  .table-shell {
    max-height: 420px;
  }
}

@media (max-width: 760px) {
  .app-shell {
    width: min(100% - 24px, 640px);
    padding-top: 16px;
  }

  .hero-copy,
  .upload-card,
  .panel-head {
    padding: 18px;
  }

  h1 {
    font-size: 28px;
  }

  .upload-card,
  .metric-grid,
  .example-grid,
  .ask-form,
  .chart-grid {
    grid-template-columns: 1fr;
  }
}
</style>
