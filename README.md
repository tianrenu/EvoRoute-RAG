# EvoRoute-RAG

> Agentic RAG Self-Evolution Framework for Campus Knowledge Services

**EvoRoute-RAG** 是一个面向校园知识服务的 **Agentic RAG 自进化系统**。核心创新是将错误反馈闭环与技能层动态优化相结合，实现检索决策的可进化演进。

---

## 核心特性

### 🔀 三层分流架构

| 层级 | 名称 | 描述 | LLM 调用 |
|------|------|------|---------|
| **L1** | 技能路由层 | Dict + Aho-Corasick 六维度 SkillScore 评分，零 LLM 调用，毫秒级响应 | ❌ |
| **L2** | Agentic RAG | LangGraph 状态图 + Qdrant 向量检索 + Quality Gate 置信度路由 | ✅ |
| **L3** | 自进化技能层 | 错误归因 → 技能生成 → 冲突检测 → 分级验证 → 版本管理 | ✅ |

### 🧬 自进化闭环

系统通过 L3 层持续自我优化：

```
用户 query → L1 路由 → L2 检索 → LLM 生成 → 答案评估
                                    ↓
                            错误归因（L3）
                                    ↓
                    技能修改 / 新技能生成 → 入库
                                    ↓
                        下次同类 query 命中新技能
```

### 📖 技能驱动检索

每个技能（Skill）是一个 YAML 配置文件，包含：
- **触发条件**：关键词、同义词、语义类型
- **检索配置**：boost_keywords、metadata 过滤器
- **进化数据**：hit_count、success_rate、false_positive 追踪

---

## 目录结构

```
EvoRoute-RAG/
├── evoroute_rag/              # 代码根包
│   ├── layer1/                # L1 技能路由层
│   │   ├── skill_matcher.py   # 匹配引擎
│   │   ├── skill_scorer.py   # SkillScore 六维度计算
│   │   ├── skill_loader.py   # YAML 技能加载 + 同义词展开
│   │   └── aho_corasick_matcher.py
│   ├── layer2/                # L2 Agentic RAG（规划中）
│   └── layer3/                # L3 自进化层（规划中）
├── skills/                    # 种子技能库（10 个校园场景技能）
│   ├── synonym_dict.yaml      # 同义词词典 v1.1
│   ├── library_hours_001.yaml
│   ├── campus_card_topup_001.yaml
│   └── ...
├── tests/
│   └── test_layer1/          # L1 单元测试（36 个用例）
├── docs/
│   ├── 架构设计说明书.md      # 详细技术设计文档
│   └── 架构解析手册_for天认u.md
└── requirements.txt
```

---

## 快速开始

### 环境要求

- Python 3.10+
- Qdrant（用于 L2 向量检索）

### 安装

```bash
git clone https://github.com/tianrenu/EvoRoute-RAG.git
cd EvoRoute-RAG
pip install -r requirements.txt
```

### 运行 L1 测试

```bash
python -m pytest tests/test_layer1/ -v
```

### 使用示例

```python
from evoroute_rag.layer1.skill_matcher import SkillMatcher

matcher = SkillMatcher(
    skill_library_path="skills",
    synonym_dict_path="skills/synonym_dict.yaml"
)

# 精确关键词匹配
result = matcher.match("图书馆今天几点开门")
print(f"命中技能: {result.skill_name}, 得分: {result.skill_score}")

# 同义词扩展匹配（"学府" → "学校"）
result = matcher.match("学府图书馆开放时间")
print(f"命中技能: {result.skill_name}, 得分: {result.skill_score}")
```

---

## 技能示例

```yaml
# skills/library_hours_001.yaml
id: library_hours_001
name: 图书馆开放时间查询
version: 1
status: active
trigger:
  keywords: [图书馆, 开放时间, 开馆, 闭馆, 营业时间]
  semantic_type: information_query
match_threshold: 0.25
answer_type: template
answer_template: |
  图书馆开放时间为：
  - 周一至周五：8:00 - 22:00
  - 周末：9:00 - 17:00
  - 节假日：详见图书馆官网公告
evolution:
  hit_count: 0
  success_rate: 0.85
```

---

## 项目阶段

| 阶段 | 目标 | 状态 |
|------|------|------|
| **Phase 1** | L1 + L2 基础闭环跑通，10 个种子技能 | 🔄 进行中 |
| **Phase 2** | L3 自进化机制，语义增强，向量检索升级 | ⏳ 待开始 |

---

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| L1 匹配 | Python Dict + Aho-Corasick + jieba 分词 |
| L2 编排 | LangGraph |
| 向量数据库 | Qdrant |
| LLM | MiniMax API（OpenAI 兼容格式） |
| 评估框架 | RAGAS |
| 技能存储 | YAML + Git 版本管理 |

---

## License

Apache License 2.0
