# 任务：开发 EvoRoute-RAG 第一层（L1）：可解释技能路由层

## 一、项目背景

你正在开发 EvoRoute-RAG，一个三层分流自进化的 Agentic RAG 系统。你的任务是实现第一层（L1）。

L1 是整个系统的"高速公路"——用确定性算法（Dict + Aho-Corasick）对用户查询做毫秒级路由，匹配到合适的"技能"后决定是直接返回答案还是传递检索指令给 L2。

**L1 的核心设计原则：**
1. **零 LLM 调用**：纯符号计算，不调用任何大模型
2. **毫秒级响应**：目标 < 10ms
3. **可解释路由**：每个匹配决策都有六维度 SkillScore 评分
4. **Phase 1 封闭技能库**：只修改已有技能字段，不新增正式 skill ID

**项目目录结构（请遵循）：**

```
EvoRoute-RAG/
├── evoroute_rag/
│ ├── __init__.py
│ ├── layer1/ # ← 你负责这里
│ │   ├── __init__.py
│ │   ├── skill_matcher.py # 技能匹配引擎（主入口）
│ │   ├── skill_scorer.py # SkillScore 六维度计算
│ │   ├── skill_loader.py # YAML 技能文件加载
│ │   └── aho_corasick_matcher.py # Aho-Corasick 多模式匹配
│ ├── layer2/ # L2 由其他模块负责
│ ├── layer3/ # L3 由其他模块负责
│ └── config/
│ └── system_config.yaml
├── skills/ # 技能库（YAML 文件）
├── tests/
│ └── test_layer1/
├── config/
└── requirements.txt
```

**你只需负责 `evoroute_rag/layer1/` 内的所有文件（4个核心模块 + __init__.py），以及 `skills/` 目录下的种子技能文件（10个 YAML 文件）。不涉及 L2 和 L3。**


## 二、技术要求

### 2.1 YAML 技能文件格式

每个技能文件存放在 `skills/` 目录下，文件名格式：`skills/{skill_id}.yaml`

```yaml
# skills/library_overdue_001.yaml
id: library_overdue_001
name: 图书馆借书超期处理
version: 1
status: active
created_at: 2026-04-10

trigger:
  keywords: [借书, 超期, 罚款, 归还, 借阅]
  aliases: [借阅图书, 图书归还]
  semantic_type: procedure_query
  question_type: 流程类

semantic_profile:
  example_queries:
    - 借书超期了怎么办
    - 图书馆超期罚款标准
  applicable_scenarios:
    - 学生归还图书时发现已超期
  response_guidelines: 简洁明了，直接给出处理步骤

match_threshold: 0.6

answer_type: template # 可选: template / directive
answer_template: |
  图书馆借书超期处理流程如下：
  1. 登录图书馆官网或微信小程序
  2. 进入"我的借阅"查看超期书籍
  3. 按超期天数缴纳罚款（0.1元/天/本）
  4. 缴纳完成后即可恢复借阅权限

action:
  retrieval:
    boost_keywords: [图书馆借阅规则, 超期处理办法]
    filter_metadata: {category: library, school_year: current}
    top_k: 5

evolution:
  hit_count: 0
  success_rate: 0.0
  false_positive_count: 0
  depends_on: []
  last_evolution_time: null
```

### 2.2 技能状态与生命周期

每个技能有 `status` 字段：
- `active`：全量参与 L1 路由
- `candidate`：L3 生成的新技能，不参与路由
- `dormant`：连续30天未使用，退出主动匹配
- `deprecated`：永久退出匹配

**skill_loader 加载时，只加载 `status: active` 的技能参与匹配。** 其他状态打印 info 日志但不参与路由。

### 2.3 SkillScore 六维度评分公式

```
SkillScore = (Σ(wi × Di) / Σwi) × D_cold
```

| 维度 | 变量 | 计算方式 | 权重 |
|------|------|---------|------|
| 关键词命中率 | D_keyword | `len(query_tokens ∩ keywords) / len(keywords)` | 0.25 |
| 模式匹配分数 | D_pattern | 归一化 Aho-Corasick 命中得分（见 2.4） | 0.20 |
| 语义类型匹配度 | D_semtype | `0.5`（L1 无法判断类型，默认值） | 0.20 |
| 同义词扩展覆盖率 | D_alias | `len(query_tokens ∩ aliases) / len(aliases)`，空时为 0 | 0.15 |
| 历史成功率 | D_hist | `skill.evolution.success_rate`，冷启动时为 0.5 | 0.10 |
| 冷启动惩罚 | D_cold | `0.8` if hit_count < 10 else `1.0`（乘性系数，不参与加权求和） | — |

**触发条件：** query 被 Aho-Corasick 部分匹配（非 Dict 精确匹配）时计算 SkillScore。

**分母为 0 的处理：** 若某维度命中数/总数均为 0，该维度得分记为 0，不导致除零错误。

**决策规则：**
- SkillScore ≥ match_threshold → 命中
- SkillScore < match_threshold → 未命中，进入 L2

### 2.4 D_pattern 计算细节

```python
def calc_d_pattern(query: str, matched_keywords: list[str], skill) -> float:
    """
    D_pattern = 所有命中关键词的 boost 总和 / 候选技能最大可能的 boost 总和
    最大可能 = 该技能 keywords + aliases 所有词的 boost 之和（归一化到 0~1）
    
    归一化公式：D_pattern = sum(matched_boosts) / sum(all_possible_boosts)
    """
    all_patterns = skill.trigger.keywords + skill.trigger.aliases
    max_possible_boost = sum(
        skill.trigger.confidence_boost for _ in all_patterns
    )
    if max_possible_boost == 0:
        return 0.0
    matched_boost = sum(
        skill.trigger.confidence_boost
        for kw in matched_keywords
        if kw in all_patterns
    )
    return matched_boost / max_possible_boost
```

每个关键词/别名在 YAML 中的 `confidence_boost` 默认为 `0.1`，可覆盖。

### 2.5 匹配流程

```python
class SkillMatcher:
    def __init__(self, skill_library_path: str):
        """加载所有 YAML 技能文件，构建 Aho-Corasick 自动机"""
        self.skills = self._load_skills(skill_library_path)  # 仅 status=active
        self.ac_automaton = self._build_aho_corasick()
        self._build_keyword_dict()  # Dict 精确匹配用

    def match(self, query: str) -> Optional[SkillMatchResult]:
        """
        匹配流程：
        1. Dict 精确匹配：query_token 完全命中某技能的 keywords → 直接返回（最高优先级）
        2. Aho-Corasick 多模式匹配：query 部分命中多个技能的 keywords + aliases
        3. 对所有 Aho-Corasick 命中的候选技能，逐个计算 SkillScore
        4. 取 SkillScore 最高且 ≥ match_threshold 的技能
        5. 若无技能达标，返回 None（进入 L2）
        """
        # Step 1: Dict 精确匹配
        dict_result = self._dict_match(query)
        if dict_result:
            return dict_result

        # Step 2: Aho-Corasick 批量匹配
        ac_matches = self._aho_corasick_match(query)
        if not ac_matches:
            return None  # 没有任何命中，进入 L2

        # Step 3: 逐个计算 SkillScore
        candidates = []
        for skill_id, matched_keywords in ac_matches.items():
            skill = self._get_skill(skill_id)
            score, breakdown = self.scorer.calc_skill_score(
                query, skill, matched_keywords
            )
            if score >= skill.match_threshold:
                candidates.append((skill, score, breakdown))

        if not candidates:
            return None  # 无人达标，进入 L2

        # Step 4: 返回最高分
        best = max(candidates, key=lambda x: x[1])
        return self._build_result(best[0], best[1], best[2], "aho_corasick")
```

### 2.6 SkillMatchResult 数据结构

```python
@dataclass
class SkillMatchResult:
    skill_id: str
    skill_name: str
    answer_type: str              # "template" 或 "directive"
    answer_template: Optional[str]  # answer_type=template 时有值
    retrieval_config: Optional[dict] # answer_type=directive 时有值
    skill_score: float
    score_breakdown: dict        # 六维度明细，用于可解释性
    matched_by: str              # "dict" 或 "aho_corasick"
```

### 2.7 Aho-Corasick 自动机构建

- 使用 `pyahocorasick` 库（需加入 requirements.txt）
- 构建时，每条模式（keyword/alias）关联其 `skill_id`
- 匹配时返回：`{skill_id: [matched_keyword1, matched_keyword2, ...]}`

### 2.8 Dict 精确匹配规则

- 对 query 分词后的每个 token，检查是否完整等于某技能的某个 keyword
- 不做子串匹配，只做完整词匹配
- 命中则立即返回，不继续匹配其他技能（取第一个匹配）


## 三、需要创建的种子技能（10个 YAML 文件）

请在 `skills/` 目录下创建以下 10 个种子技能文件：

| skill_id | name | question_type | answer_type |
|----------|------|--------------|-------------|
| library_overdue_001 | 图书馆借书超期处理 | 流程指引类 | template |
| library_hours_001 | 图书馆开放时间查询 | 事实查询类 | template |
| scholarship_time_001 | 奖学金申请时间查询 | 事实查询类 | template |
| baoyan_qualification_001 | 保研资格条件查询 | 资格判断类 | directive |
| gpa_calculation_001 | GPA绩点计算方法 | 数值计算类 | template |
| course_add_drop_001 | 选课补退选流程 | 流程指引类 | template |
| poverty_identification_001 | 贫困生认定申请流程 | 流程指引类 | template |
| financial_aid_qualification_001 | 助学贷款申请资格 | 资格判断类 | directive |
| grade_appeal_001 | 成绩复核申请流程 | 流程指引类 | template |
| campus_card_topup_001 | 校园卡充值方法 | 事实查询类 | template |

**每个技能的 keywords 和 aliases 需要根据技能内容合理设计：**
- keywords：5个以上中文核心词
- aliases：2-3个同义词/表达变体
- `trigger.confidence_boost: 0.1`（每个关键词默认 boost 值）

**answer_template 必须写清楚具体内容**（不要留空或写"待补充"）。


## 四、关键实现要求

1. **所有代码必须加类型注解**（Python typing）
2. **所有函数必须加 docstring**（Google 风格）
3. **skill_loader 必须处理 YAML 格式错误**：打印 warning，跳过损坏文件，不中断整个加载
4. **SkillScore 计算必须输出每个维度的明细**（用于可解释性，方便调试）
5. **skill_scorer 边界情况处理**：
   - aliases 为空时 D_alias = 0
   - keywords 为空时 D_keyword = 0
   - 分母为 0 时该维度得分为 0
   - hit_count < 10 时 D_cold = 0.8，否则 = 1.0
   - success_rate 为 None 或空时默认为 0.5
6. **不调用任何 LLM**（L1 是纯符号计算）
7. **不调用 embedding 模型**（Phase 1 不做语义召回）


## 五、依赖库

请在 requirements.txt 中加入：
```
pyahocorasick>=2.0.0
pyyaml>=6.0
jieba>=0.42.1
```

不要加入 langchain、qdrant-client、ragas 等 L2/L3 的依赖。

## 六、验收标准（开发完成后请自行验证）

1. ✅ 10 个技能文件正常加载（仅 status=active），无报错
2. ✅ Dict 精确匹配："借书超期了怎么办" 命中 library_overdue_001（matched_by="dict"）
3. ✅ Aho-Corasick 部分匹配 + SkillScore 评分："超期罚款怎么交" 命中 library_overdue_001（matched_by="aho_corasick"）
4. ✅ 非相关问题（如"今天天气怎么样"）返回 None（不命中任何技能）
5. ✅ SkillScore 每个维度都有具体的数值输出（不是全部 0 或 1）
6. ✅ 响应时间 < 10ms（用 time.perf_counter 测量 10 次取均值）
7. ✅ answer_type=directive 的技能返回 retrieval_c

## 七、交付物清单

1. `evoroute_rag/layer1/__init__.py` - 模块入口，导出 SkillMatcher
2. `evoroute_rag/layer1/skill_matcher.py` - 技能匹配引擎（主入口类）
3. `evoroute_rag/layer1/skill_scorer.py` - SkillScore 六维度计算
4. `evoroute_rag/layer1/skill_loader.py` - YAML 技能文件加载器
5. `evoroute_rag/layer1/aho_corasick_matcher.py` - Aho-Corasick 多模式匹配封装
6. `evoroute_rag/__init__.py` - 包入口
7. `skills/` 目录下 10 个 YAML 文件
8. `tests/test_layer1/test_skill_loader.py`
9. `tests/test_layer1/test_skill_scorer.py`
10. `tests/test_layer1/test_skill_matcher.py`
11. `requirements.txt` - 仅包含 L1 需要的依赖

请在完成所有文件和验收标准验证后，提供每个文件的完整代码和简要说明。