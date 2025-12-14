# 渐进式工具加载解决方案

> **Progressive Tool Loading Solution for Strands Agent Framework**

本文档提供了一套完整的渐进式工具加载方案，用于在 Strands Agent 框架中实现"按需动态注入工具"的能力。

---

## 目录

- [背景与目标](#背景与目标)
- [框架能力分析](#框架能力分析)
- [解决方案架构](#解决方案架构)
- [核心实现](#核心实现)
- [使用示例](#使用示例)
- [高级配置](#高级配置)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)

---

## 背景与目标

### 问题描述

在复杂的 Agent 应用中，工具数量可能达到数十甚至上百个。将所有工具的 Schema 一次性注入模型会导致：

1. **上下文噪点过高**：大量无关工具干扰模型决策
2. **Token 消耗增加**：工具 Schema 占用宝贵的上下文窗口
3. **决策质量下降**：选项过多导致模型选择困难

### 解决目标

实现**渐进式工具加载**机制：

```
┌────────────────────────────────────────────────────────────────────┐
│  P1: 初始阶段                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  仅注入"必选工具 A" → 模型被强制"只能走这条路"                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              ↓                                      │
│  P2: 动态解锁                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  根据工具 A 的返回结果 → 代码判断并注入工具 B/C                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              ↓                                      │
│  P3: 最低噪点                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  历史结果 + 当前阶段工具 Schema → 上下文始终维持最低噪点水平       │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 框架能力分析

### 核心组件支持

经过对 Strands Agent 框架的深入分析，确认以下组件完全支持渐进式工具加载：

| 组件 | 文件位置 | 关键能力 |
|------|----------|----------|
| `ToolRegistry` | `src/strands/tools/registry.py` | 动态注册/注销工具 |
| `HookRegistry` | `src/strands/hooks/registry.py` | 生命周期事件拦截 |
| `AgentState` | `src/strands/agent/state.py` | 跨轮次状态持久化 |
| `event_loop_cycle` | `src/strands/event_loop/event_loop.py` | 模型调用前工具获取 |

### 关键注入点

1. **模型调用前获取工具列表**（[event_loop.py:335-339](file:///Users/zhubingjian/awsome-proj/python-proj/sdk-python/src/strands/event_loop/event_loop.py#L335-L339)）：

```python
if structured_output_context.forced_mode:
    tool_spec = structured_output_context.get_tool_spec()
    tool_specs = [tool_spec] if tool_spec else []
else:
    tool_specs = agent.tool_registry.get_all_tool_specs()  # ⭐ 注入点
```

2. **Hook 事件触发顺序**：

```
BeforeModelCallEvent → Model Inference → AfterModelCallEvent
                                              ↓
                                        [if tool_use]
                                              ↓
BeforeToolCallEvent → Tool Execution → AfterToolCallEvent
                                              ↓
                                    [recurse_event_loop]
```

### Hook 事件详解

| 事件 | 触发时机 | 可操作性 | 推荐用途 |
|------|----------|----------|----------|
| `BeforeModelCallEvent` | 模型调用前 | 只读 | ⭐ 动态调整工具列表 |
| `AfterModelCallEvent` | 模型返回后 | 可获取结果 | 记录/分析 |
| `BeforeToolCallEvent` | 工具执行前 | 可修改/取消/中断 | 权限控制 |
| `AfterToolCallEvent` | 工具执行后 | 可修改结果 | ⭐ 根据结果解锁新工具 |

---

## 解决方案架构

### 整体架构图

```
                         ┌─────────────────────────────────┐
                         │           用户输入              │
                         └───────────────┬─────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────────────┐
│                            Agent                                    │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                    ProgressiveToolLoader                    │    │
│  │                      (HookProvider)                         │    │
│  │  ┌────────────────────────────────────────────────────┐    │    │
│  │  │  tool_phases: 工具阶段定义                          │    │    │
│  │  │  unlock_rules: 解锁规则配置                         │    │    │
│  │  │  current_tools: 当前可用工具集                      │    │    │
│  │  └────────────────────────────────────────────────────┘    │    │
│  └────────────────────────────────────────────────────────────┘    │
│                              │                                      │
│  ┌───────────────────────────┴───────────────────────────┐         │
│  │                                                        │         │
│  ▼                                                        ▼         │
│  ┌──────────────────────┐    ┌───────────────────────────┐         │
│  │ on_before_model_call │    │   on_after_tool_call      │         │
│  │                      │    │                           │         │
│  │ - 读取 agent.state   │    │ - 分析工具执行结果         │         │
│  │ - 确定当前阶段       │    │ - 检查解锁条件             │         │
│  │ - 注入对应工具       │    │ - 更新 agent.state        │         │
│  │ - 清理过期工具       │    │ - 记录工具使用历史         │         │
│  └──────────┬───────────┘    └───────────────┬───────────┘         │
│             │                                 │                     │
│             ▼                                 ▼                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                      ToolRegistry                           │    │
│  │   ┌─────────────────────────────────────────────────────┐  │    │
│  │   │ registry: Dict[str, AgentTool]  # 当前可用工具       │  │    │
│  │   └─────────────────────────────────────────────────────┘  │    │
│  │                                                             │    │
│  │   register_tool() / register_dynamic_tool()                 │    │
│  │   get_all_tool_specs() → 返回给模型的工具列表               │    │
│  └────────────────────────────────────────────────────────────┘    │
│                              │                                      │
└──────────────────────────────┼──────────────────────────────────────┘
                               ▼
                  ┌─────────────────────────────┐
                  │     Model (LLM)             │
                  │  只看到当前阶段的工具        │
                  └─────────────────────────────┘
```

### 状态流转

```
┌──────────────────────────────────────────────────────────────────┐
│                        状态机流转图                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐      analyze_intent       ┌─────────────────┐ │
│   │   initial   │ ────────返回───────────▶  │   intent_known  │ │
│   │             │      intent类型            │                 │ │
│   │ 工具: [A]   │                           │ 工具: [A, B/C]  │ │
│   └─────────────┘                           └────────┬────────┘ │
│                                                      │          │
│                                                      │ 工具执行  │
│                                                      ▼          │
│                                             ┌─────────────────┐ │
│                                             │   task_done     │ │
│                                             │                 │ │
│                                             │ 工具: [A]       │ │
│                                             │ (重置为初始)     │ │
│                                             └─────────────────┘ │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 核心实现

### 1. 工具定义

首先定义所有可能用到的工具：

```python
# tools/intent_analyzer.py
from strands import tool

@tool
def analyze_intent(user_input: str) -> dict:
    """分析用户意图，确定下一步需要的工具。
    
    这是入口工具，所有对话都应该先经过意图分析。
    
    Args:
        user_input: 用户的原始输入文本
        
    Returns:
        包含意图类型和推荐解锁工具的字典
    """
    import json
    
    # 意图识别逻辑
    intent_mapping = {
        "搜索": {"intent": "search", "unlock": ["web_search", "news_search"]},
        "查询": {"intent": "query", "unlock": ["database_query", "api_query"]},
        "计算": {"intent": "compute", "unlock": ["calculator", "data_analysis"]},
        "写作": {"intent": "writing", "unlock": ["text_generator", "translator"]},
        "图像": {"intent": "image", "unlock": ["image_generator", "image_analyzer"]},
    }
    
    for keyword, config in intent_mapping.items():
        if keyword in user_input:
            return json.dumps(config, ensure_ascii=False)
    
    return json.dumps({"intent": "general", "unlock": []}, ensure_ascii=False)
```

```python
# tools/search_tools.py
from strands import tool

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """执行网络搜索。
    
    Args:
        query: 搜索关键词
        max_results: 最大返回结果数
        
    Returns:
        搜索结果的JSON字符串
    """
    # 实际实现中调用搜索API
    return f"搜索 '{query}' 的结果..."

@tool
def news_search(topic: str, days: int = 7) -> str:
    """搜索最新新闻。
    
    Args:
        topic: 新闻主题
        days: 搜索最近几天的新闻
        
    Returns:
        新闻列表的JSON字符串
    """
    return f"关于 '{topic}' 的最新新闻..."
```

```python
# tools/compute_tools.py
from strands import tool

@tool
def calculator(expression: str) -> str:
    """执行数学计算。
    
    Args:
        expression: 数学表达式
        
    Returns:
        计算结果
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

@tool
def data_analysis(data: list, operation: str) -> str:
    """执行数据分析操作。
    
    Args:
        data: 数据列表
        operation: 分析操作类型 (mean/sum/max/min)
        
    Returns:
        分析结果
    """
    operations = {
        "mean": lambda d: sum(d) / len(d),
        "sum": sum,
        "max": max,
        "min": min,
    }
    if operation in operations:
        return str(operations[operation](data))
    return f"不支持的操作: {operation}"
```

### 2. 渐进式工具加载器（核心组件）

```python
# progressive_loader/loader.py
"""
渐进式工具加载器 - Progressive Tool Loader

实现按需动态注入工具的核心组件。
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import (
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    AgentInitializedEvent,
)
from strands.types.tools import AgentTool

logger = logging.getLogger(__name__)


@dataclass
class ToolPhase:
    """工具阶段定义"""
    name: str                           # 阶段名称
    tools: List[str]                    # 该阶段可用的工具名称列表
    description: str = ""               # 阶段描述


@dataclass
class UnlockRule:
    """工具解锁规则"""
    trigger_tool: str                   # 触发工具名称
    condition: Callable[[dict], bool]   # 解锁条件函数
    unlock_tools: List[str]             # 解锁的工具列表
    target_phase: Optional[str] = None  # 目标阶段（可选）


@dataclass
class LoaderState:
    """加载器状态"""
    current_phase: str = "initial"
    unlocked_tools: Set[str] = field(default_factory=set)
    tool_history: List[dict] = field(default_factory=list)
    phase_history: List[str] = field(default_factory=list)


class ProgressiveToolLoader(HookProvider):
    """渐进式工具加载器
    
    通过 Hook 机制动态管理 Agent 可用的工具集，实现：
    1. 初始阶段只注入必选工具
    2. 根据工具执行结果动态解锁新工具
    3. 保持上下文最低噪点
    
    Attributes:
        tool_pool: 所有可用工具的注册池
        phases: 工具阶段定义
        unlock_rules: 解锁规则列表
        initial_phase: 初始阶段名称
    
    Example:
        ```python
        loader = ProgressiveToolLoader(
            tool_pool={
                "analyze_intent": analyze_intent_tool,
                "web_search": web_search_tool,
            },
            phases=[
                ToolPhase("initial", ["analyze_intent"]),
                ToolPhase("search", ["analyze_intent", "web_search"]),
            ],
            unlock_rules=[
                UnlockRule(
                    trigger_tool="analyze_intent",
                    condition=lambda r: "search" in r.get("intent", ""),
                    unlock_tools=["web_search"],
                    target_phase="search",
                ),
            ],
        )
        
        agent = Agent(tools=[], hooks=[loader])
        ```
    """
    
    def __init__(
        self,
        tool_pool: Dict[str, AgentTool],
        phases: List[ToolPhase],
        unlock_rules: List[UnlockRule],
        initial_phase: str = "initial",
        auto_reset: bool = True,
    ):
        """初始化加载器
        
        Args:
            tool_pool: 工具名称到工具实例的映射
            phases: 工具阶段定义列表
            unlock_rules: 解锁规则列表
            initial_phase: 初始阶段名称
            auto_reset: 任务完成后是否自动重置到初始阶段
        """
        self.tool_pool = tool_pool
        self.phases = {p.name: p for p in phases}
        self.unlock_rules = unlock_rules
        self.initial_phase = initial_phase
        self.auto_reset = auto_reset
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置一致性"""
        # 检查初始阶段是否存在
        if self.initial_phase not in self.phases:
            raise ValueError(f"初始阶段 '{self.initial_phase}' 未在 phases 中定义")
        
        # 检查所有阶段的工具是否在工具池中
        for phase in self.phases.values():
            for tool_name in phase.tools:
                if tool_name not in self.tool_pool:
                    raise ValueError(
                        f"阶段 '{phase.name}' 引用的工具 '{tool_name}' 未在 tool_pool 中注册"
                    )
        
        # 检查解锁规则的工具是否在工具池中
        for rule in self.unlock_rules:
            for tool_name in rule.unlock_tools:
                if tool_name not in self.tool_pool:
                    raise ValueError(
                        f"解锁规则引用的工具 '{tool_name}' 未在 tool_pool 中注册"
                    )
    
    def register_hooks(self, registry: HookRegistry, **kwargs):
        """注册 Hook 回调"""
        registry.add_callback(AgentInitializedEvent, self._on_agent_initialized)
        registry.add_callback(BeforeModelCallEvent, self._on_before_model_call)
        registry.add_callback(AfterToolCallEvent, self._on_after_tool_call)
    
    def _on_agent_initialized(self, event: AgentInitializedEvent):
        """Agent 初始化时设置初始状态"""
        agent = event.agent
        
        # 初始化加载器状态
        if "progressive_loader" not in agent.state:
            agent.state["progressive_loader"] = LoaderState(
                current_phase=self.initial_phase,
                unlocked_tools=set(self.phases[self.initial_phase].tools),
            ).__dict__
        
        # 注入初始工具
        self._inject_tools(agent)
        
        logger.info(
            f"ProgressiveToolLoader 初始化完成, "
            f"阶段: {self.initial_phase}, "
            f"工具: {self.phases[self.initial_phase].tools}"
        )
    
    def _on_before_model_call(self, event: BeforeModelCallEvent):
        """模型调用前更新可用工具"""
        agent = event.agent
        state = self._get_state(agent)
        
        # 根据当前阶段和已解锁工具更新工具注册表
        self._inject_tools(agent)
        
        current_tools = list(state["unlocked_tools"])
        logger.debug(f"模型调用前工具列表: {current_tools}")
    
    def _on_after_tool_call(self, event: AfterToolCallEvent):
        """工具执行后检查解锁条件"""
        agent = event.agent
        tool_name = event.tool_use.get("name")
        result = event.result
        
        state = self._get_state(agent)
        
        # 记录工具使用历史
        state["tool_history"].append({
            "tool": tool_name,
            "input": event.tool_use.get("input"),
            "result_status": result.get("status"),
        })
        
        # 检查解锁规则
        self._check_unlock_rules(agent, tool_name, result)
        
        # 持久化状态
        self._save_state(agent, state)
    
    def _check_unlock_rules(self, agent, tool_name: str, result: dict):
        """检查并应用解锁规则"""
        state = self._get_state(agent)
        
        for rule in self.unlock_rules:
            if rule.trigger_tool != tool_name:
                continue
            
            # 解析工具返回结果
            parsed_result = self._parse_tool_result(result)
            
            # 检查解锁条件
            try:
                if rule.condition(parsed_result):
                    # 解锁新工具
                    new_tools = set(rule.unlock_tools) - state["unlocked_tools"]
                    if new_tools:
                        state["unlocked_tools"].update(new_tools)
                        logger.info(f"解锁新工具: {new_tools}")
                    
                    # 切换阶段
                    if rule.target_phase:
                        old_phase = state["current_phase"]
                        state["current_phase"] = rule.target_phase
                        state["phase_history"].append(old_phase)
                        logger.info(f"阶段切换: {old_phase} → {rule.target_phase}")
                    
                    # 立即更新工具注册表
                    self._inject_tools(agent)
                    
            except Exception as e:
                logger.warning(f"检查解锁条件时出错: {e}")
    
    def _parse_tool_result(self, result: dict) -> dict:
        """解析工具返回结果"""
        if result.get("status") != "success":
            return {"status": "error", "error": str(result)}
        
        content = result.get("content", [])
        if content and "text" in content[0]:
            try:
                return json.loads(content[0]["text"])
            except json.JSONDecodeError:
                return {"text": content[0]["text"]}
        
        return result
    
    def _inject_tools(self, agent):
        """注入当前阶段的工具到 agent"""
        state = self._get_state(agent)
        
        # 清空现有工具
        agent.tool_registry.registry.clear()
        
        # 注入已解锁的工具
        for tool_name in state["unlocked_tools"]:
            if tool_name in self.tool_pool:
                tool = self.tool_pool[tool_name]
                agent.tool_registry.register_tool(tool)
    
    def _get_state(self, agent) -> dict:
        """获取加载器状态"""
        if "progressive_loader" not in agent.state:
            agent.state["progressive_loader"] = LoaderState(
                current_phase=self.initial_phase,
                unlocked_tools=set(self.phases[self.initial_phase].tools),
            ).__dict__
        
        state = agent.state["progressive_loader"]
        
        # 确保 unlocked_tools 是 set 类型
        if isinstance(state.get("unlocked_tools"), list):
            state["unlocked_tools"] = set(state["unlocked_tools"])
        
        return state
    
    def _save_state(self, agent, state: dict):
        """保存加载器状态"""
        # 转换 set 为 list 以便序列化
        save_state = state.copy()
        save_state["unlocked_tools"] = list(state["unlocked_tools"])
        agent.state["progressive_loader"] = save_state
    
    def reset(self, agent):
        """重置到初始状态"""
        agent.state["progressive_loader"] = LoaderState(
            current_phase=self.initial_phase,
            unlocked_tools=set(self.phases[self.initial_phase].tools),
        ).__dict__
        self._inject_tools(agent)
        logger.info(f"加载器已重置到初始阶段: {self.initial_phase}")
```

### 3. 简化版加载器（快速使用）

```python
# progressive_loader/simple_loader.py
"""简化版渐进式工具加载器 - 适用于简单场景"""

from typing import Callable, Dict, List
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import BeforeModelCallEvent, AfterToolCallEvent
from strands.types.tools import AgentTool


class SimpleProgressiveLoader(HookProvider):
    """简化版渐进式工具加载器
    
    Example:
        ```python
        loader = SimpleProgressiveLoader(
            initial_tools=[analyze_intent],
            tool_pool={
                "web_search": web_search,
                "calculator": calculator,
            },
        )
        
        agent = Agent(tools=[], hooks=[loader])
        ```
    """
    
    def __init__(
        self,
        initial_tools: List[AgentTool],
        tool_pool: Dict[str, AgentTool],
    ):
        self.initial_tools = {t.tool_name: t for t in initial_tools}
        self.tool_pool = tool_pool
        self.all_tools = {**self.initial_tools, **tool_pool}
    
    def register_hooks(self, registry: HookRegistry, **kwargs):
        registry.add_callback(BeforeModelCallEvent, self._before_model)
        registry.add_callback(AfterToolCallEvent, self._after_tool)
    
    def _before_model(self, event: BeforeModelCallEvent):
        """模型调用前注入工具"""
        agent = event.agent
        unlocked = agent.state.get("unlocked_tools", set(self.initial_tools.keys()))
        
        agent.tool_registry.registry.clear()
        for name in unlocked:
            if name in self.all_tools:
                agent.tool_registry.register_tool(self.all_tools[name])
    
    def _after_tool(self, event: AfterToolCallEvent):
        """工具执行后解析解锁指令"""
        agent = event.agent
        result = event.result
        
        if result.get("status") != "success":
            return
        
        # 尝试从结果中提取 unlock 字段
        try:
            import json
            content = result.get("content", [])
            if content and "text" in content[0]:
                data = json.loads(content[0]["text"])
                if "unlock" in data:
                    current = agent.state.get("unlocked_tools", set(self.initial_tools.keys()))
                    if isinstance(current, list):
                        current = set(current)
                    current.update(data["unlock"])
                    agent.state["unlocked_tools"] = current
        except:
            pass
```

---

## 使用示例

### 基础用法

```python
from strands import Agent
from progressive_loader import ProgressiveToolLoader, ToolPhase, UnlockRule

# 定义工具
from tools.intent_analyzer import analyze_intent
from tools.search_tools import web_search, news_search
from tools.compute_tools import calculator, data_analysis

# 创建工具池
tool_pool = {
    "analyze_intent": analyze_intent,
    "web_search": web_search,
    "news_search": news_search,
    "calculator": calculator,
    "data_analysis": data_analysis,
}

# 定义阶段
phases = [
    ToolPhase("initial", ["analyze_intent"], "初始阶段 - 仅意图分析"),
    ToolPhase("search", ["analyze_intent", "web_search", "news_search"], "搜索阶段"),
    ToolPhase("compute", ["analyze_intent", "calculator", "data_analysis"], "计算阶段"),
]

# 定义解锁规则
unlock_rules = [
    UnlockRule(
        trigger_tool="analyze_intent",
        condition=lambda r: r.get("intent") == "search",
        unlock_tools=["web_search", "news_search"],
        target_phase="search",
    ),
    UnlockRule(
        trigger_tool="analyze_intent",
        condition=lambda r: r.get("intent") == "compute",
        unlock_tools=["calculator", "data_analysis"],
        target_phase="compute",
    ),
]

# 创建加载器
loader = ProgressiveToolLoader(
    tool_pool=tool_pool,
    phases=phases,
    unlock_rules=unlock_rules,
)

# 创建 Agent
agent = Agent(
    system_prompt="""你是一个智能助手。
    
对于每个用户请求，你需要：
1. 首先使用 analyze_intent 分析用户意图
2. 根据意图使用相应的工具完成任务

注意：初始时你只能使用 analyze_intent 工具，其他工具会根据意图分析结果自动解锁。
""",
    tools=[],  # 空工具列表，由加载器管理
    hooks=[loader],
)

# 使用 Agent
result = agent("帮我搜索 Python 最新版本")
print(result)
```

### 简化版用法

```python
from strands import Agent
from progressive_loader import SimpleProgressiveLoader
from tools.intent_analyzer import analyze_intent
from tools.search_tools import web_search
from tools.compute_tools import calculator

# 创建简化版加载器
loader = SimpleProgressiveLoader(
    initial_tools=[analyze_intent],
    tool_pool={
        "web_search": web_search,
        "calculator": calculator,
    },
)

agent = Agent(tools=[], hooks=[loader])
result = agent("计算 123 * 456")
```

---

## 高级配置

### 多级解锁

```python
# 支持多级工具解锁链
unlock_rules = [
    # 第一级：意图分析 → 解锁一级工具
    UnlockRule(
        trigger_tool="analyze_intent",
        condition=lambda r: r.get("intent") == "complex_query",
        unlock_tools=["query_planner"],
        target_phase="planning",
    ),
    # 第二级：查询规划 → 解锁二级工具
    UnlockRule(
        trigger_tool="query_planner",
        condition=lambda r: r.get("requires_database", False),
        unlock_tools=["database_query", "result_formatter"],
        target_phase="execution",
    ),
]
```

### 条件组合

```python
def complex_condition(result: dict) -> bool:
    """复杂解锁条件"""
    return (
        result.get("intent") == "data_processing" 
        and result.get("data_size", 0) > 1000
        and "historical" in result.get("data_type", "")
    )

unlock_rules = [
    UnlockRule(
        trigger_tool="data_analyzer",
        condition=complex_condition,
        unlock_tools=["batch_processor", "async_handler"],
    ),
]
```

### 与会话管理集成

```python
from strands.session import FileSessionManager

# 加载器状态会自动通过 agent.state 持久化
agent = Agent(
    tools=[],
    hooks=[loader],
    session_manager=FileSessionManager(session_dir="./sessions"),
)

# 会话恢复时，工具解锁状态也会恢复
```

---

## 最佳实践

### 1. 工具设计原则

| 原则 | 说明 |
|------|------|
| **单一职责** | 每个工具只做一件事 |
| **明确输出** | 返回结构化 JSON，包含 `unlock` 字段 |
| **幂等设计** | 相同输入产生相同输出 |
| **错误处理** | 优雅处理异常，返回有意义的错误信息 |

### 2. 阶段设计建议

```python
# ✅ 推荐：小步递进
phases = [
    ToolPhase("initial", ["analyze"]),
    ToolPhase("planning", ["analyze", "plan"]),
    ToolPhase("execution", ["analyze", "plan", "execute"]),
    ToolPhase("validation", ["analyze", "plan", "execute", "validate"]),
]

# ❌ 避免：跳跃太大
phases = [
    ToolPhase("initial", ["analyze"]),
    ToolPhase("all", ["analyze", "plan", "execute", "validate", "report", "notify"]),
]
```

### 3. 解锁条件设计

```python
# ✅ 推荐：明确、可测试的条件
UnlockRule(
    trigger_tool="classify",
    condition=lambda r: r.get("category") == "finance" and r.get("confidence", 0) > 0.8,
    unlock_tools=["stock_query"],
)

# ❌ 避免：模糊条件
UnlockRule(
    trigger_tool="classify",
    condition=lambda r: "可能" in str(r),  # 不确定的条件
    unlock_tools=["stock_query"],
)
```

### 4. 日志与监控

```python
import logging

# 启用详细日志
logging.getLogger("progressive_loader").setLevel(logging.DEBUG)

# 查看工具加载情况
# [DEBUG] 模型调用前工具列表: ['analyze_intent']
# [INFO] 解锁新工具: {'web_search', 'news_search'}
# [INFO] 阶段切换: initial → search
```

---

## 常见问题

### MCP 工具的渐进式解锁（HookProvider 草稿）

下面给出一个可直接复用的 HookProvider 草稿，适用于“初始仅暴露工具 A，执行成功后按需解锁本地工具 B 与 MCP 工具”的场景。核心逻辑：

- `BeforeModelCallEvent` 按阶段过滤工具列表，首轮只让模型看到必选工具 A，保持最低噪点。
- `AfterToolCallEvent` 检测工具 A 结果，满足条件后动态注册工具 B 与 MCP 工具，下一轮才会出现在模型的 tool specs。
- `BeforeToolCallEvent` 拦截未解锁工具的越权调用，直接返回取消消息，避免走错分支。
- `AfterInvocationEvent` 清理注册表并停止 MCP 连接，避免资源泄露。

```python
# hooks/stepwise_tool_gate.py
import asyncio
from typing import Callable

from strands.hooks import (
    HookProvider,
    HookRegistry,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    AfterToolCallEvent,
    AfterInvocationEvent,
)
from strands.types.tools import AgentTool, ToolResult
from strands.tools.mcp.mcp_client import MCPClient


class StepwiseToolGate(HookProvider):
    def __init__(
        self,
        tool_a_name: str,
        tool_b: AgentTool,
        mcp_client: MCPClient,
        should_unlock: Callable[[ToolResult], bool],
        mcp_tool_name: str | None = None,  # 指定要解锁的 MCP 工具名；None 表示取第一个
    ) -> None:
        self.tool_a_name = tool_a_name
        self.tool_b = tool_b
        self.mcp_client = mcp_client
        self.should_unlock = should_unlock
        self.mcp_tool_name = mcp_tool_name
        self._orig_registry: dict[str, AgentTool] = {}
        self._staged_mcp_tool: AgentTool | None = None

    def register_hooks(self, registry: HookRegistry, **_: dict) -> None:
        registry.add_callback(BeforeInvocationEvent, self._on_start)
        registry.add_callback(BeforeModelCallEvent, self._filter_tools)
        registry.add_callback(BeforeToolCallEvent, self._guard_pre_tool)
        registry.add_callback(AfterToolCallEvent, self._maybe_unlock)
        registry.add_callback(AfterInvocationEvent, self._cleanup)

    def _on_start(self, event: BeforeInvocationEvent) -> None:
        event.agent.state["stage"] = "a-only"
        self._orig_registry = event.agent.tool_registry.registry.copy()

    async def _filter_tools(self, event: BeforeModelCallEvent) -> None:
        stage = event.agent.state.get("stage", "a-only")
        if stage == "a-only":
            allowed = {self.tool_a_name}
            event.agent.tool_registry.registry = {
                k: v for k, v in self._orig_registry.items() if k in allowed
            }
        else:
            event.agent.tool_registry.registry = self._orig_registry

    def _guard_pre_tool(self, event: BeforeToolCallEvent) -> None:
        stage = event.agent.state.get("stage", "a-only")
        if stage == "a-only" and event.tool_use["name"] != self.tool_a_name:
            event.cancel_tool = "This tool is not yet unlocked; call tool_a first."

    async def _maybe_unlock(self, event: AfterToolCallEvent) -> None:
        if event.tool_use["name"] != self.tool_a_name:
            return
        result = event.result
        if isinstance(result, Exception):
            return
        if not self.should_unlock(result):
            return

        if self._staged_mcp_tool is None:
            tools = await self.mcp_client.load_tools()
            chosen = None
            if self.mcp_tool_name:
                chosen = next((t for t in tools if t.tool_name == self.mcp_tool_name), None)
            if chosen is None and tools:
                chosen = tools[0]
            self._staged_mcp_tool = chosen

        if self.tool_b.tool_name not in event.agent.tool_registry.registry:
            event.agent.tool_registry.register_dynamic_tool(self.tool_b)
            self._orig_registry[self.tool_b.tool_name] = self.tool_b

        if self._staged_mcp_tool and self._staged_mcp_tool.tool_name not in event.agent.tool_registry.registry:
            event.agent.tool_registry.register_dynamic_tool(self._staged_mcp_tool)
            self._orig_registry[self._staged_mcp_tool.tool_name] = self._staged_mcp_tool

        event.agent.state["stage"] = "unlocked"

    def _cleanup(self, event: AfterInvocationEvent) -> None:
        event.agent.tool_registry.registry = self._orig_registry
        event.agent.state["stage"] = "a-only"
        try:
            self.mcp_client.stop(None, None, None)  # 与 MCPClient __exit__ 一致
        except Exception:
            pass


# 运行时接入
from strands import Agent


def unlock_pred(result: ToolResult) -> bool:
    return result.get("status") == "success"


mcp_client = MCPClient(transport_callable=your_transport_factory)

agent = Agent(
    tools=[tool_a],  # 初始只暴露 A
    hooks=[StepwiseToolGate(tool_a_name="tool_a", tool_b=tool_b, mcp_client=mcp_client, should_unlock=unlock_pred)],
)
```

实施提示：

- 首轮仅注册工具 A，避免 MCP 工具 schema 干扰模型决策；待 A 成功后再注册 MCP 工具及本地 B。
- 若需跨会话持久解锁，可移除 `_cleanup` 中的阶段复位与 MCP 关闭逻辑，改为上层管理生命周期。
- 若担心模型越权，可在 `_guard_pre_tool` 中返回更明确的错误文本，引导模型重试 A。

### Q1: 工具热重载后解锁状态会丢失吗？

**A**: 不会。解锁状态存储在 `agent.state["progressive_loader"]` 中，与工具实例无关。但需要确保工具池中包含热重载后的新工具实例。

### Q2: 如何手动重置到初始状态？

**A**: 调用加载器的 `reset()` 方法：

```python
loader.reset(agent)
```

### Q3: 多个加载器可以同时使用吗？

**A**: 不推荐。如果需要更复杂的逻辑，建议扩展单个加载器的功能。

### Q4: 与结构化输出 (`structured_output`) 兼容吗？

**A**: 需要注意，使用 `structured_output` 时框架会强制使用特定工具。建议在这种场景下禁用渐进式加载，或确保结构化输出工具始终可用。

### Q5: 如何调试解锁条件？

**A**: 
1. 启用 DEBUG 日志
2. 在条件函数中添加打印语句
3. 检查 `agent.state["progressive_loader"]["tool_history"]`

---

## 附录

### API 参考

#### ProgressiveToolLoader

| 方法 | 说明 |
|------|------|
| `__init__(tool_pool, phases, unlock_rules, ...)` | 初始化加载器 |
| `register_hooks(registry)` | 注册 Hook 回调 |
| `reset(agent)` | 重置到初始状态 |

#### ToolPhase

| 属性 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 阶段名称 |
| `tools` | `List[str]` | 该阶段可用的工具 |
| `description` | `str` | 阶段描述 |

#### UnlockRule

| 属性 | 类型 | 说明 |
|------|------|------|
| `trigger_tool` | `str` | 触发工具名称 |
| `condition` | `Callable[[dict], bool]` | 解锁条件函数 |
| `unlock_tools` | `List[str]` | 解锁的工具列表 |
| `target_phase` | `Optional[str]` | 目标阶段 |

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0.0 | 2024-12-14 | 初始版本 |

---

*本文档由 Strands Agent 框架团队维护*
