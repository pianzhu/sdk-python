# Strands Agent 框架架构与工作流程

本文档详细描述了 Strands Agent 框架的核心架构、设计理念和完整工作流程。

## 目录

- [核心功能概览](#核心功能概览)
- [架构设计理念](#架构设计理念)
- [整体架构图](#整体架构图)
- [核心执行流程](#核心执行流程)
  - [流程 1: Agent 初始化](#流程-1-agent-初始化)
  - [流程 2: Agent 调用](#流程-2-agent-调用核心执行流程)
  - [流程 3: Event Loop 执行](#流程-3-event-loop-执行最核心)
  - [流程 4: 工具执行](#流程-4-工具执行)
  - [流程 5: Hook 系统](#流程-5-hook-系统)
  - [流程 6: 会话管理](#流程-6-会话管理)
  - [流程 7: 结构化输出](#流程-7-结构化输出)
  - [流程 8: 中断机制](#流程-8-中断机制)
- [设计模式与亮点](#设计模式与亮点)
- [核心文件导航](#核心文件导航)

---

## 核心功能概览

Strands Agent 框架实现了以下核心功能：

### 1. **Agent 对话能力**
- 自然语言交互：`agent("你的问题")`
- 支持多模态输入（文本、图像等）
- 流式/非流式响应
- 结构化输出（Pydantic 模型）

### 2. **工具系统**
- 动态工具注册与发现
- Python 装饰器定义工具：`@tool`
- 支持 MCP (Model Context Protocol) 工具
- 工具热重载（从 `./tools/` 目录）
- 直接工具调用：`agent.tool.tool_name(params)`

### 3. **事件驱动架构**
- Hook 系统（BeforeToolCall, AfterToolCall 等）
- 事件循环驱动的执行流程
- 支持自定义 Hook Provider

### 4. **状态与会话管理**
- Agent 状态持久化
- 会话管理器（Session Manager）
- 对话历史管理（Conversation Manager）
  - 滑动窗口策略
  - 总结策略
  - 自动上下文溢出处理

### 5. **高级特性**
- 中断机制（Interrupt）支持人在循环中
- 多 Agent 协作（Swarm, Graph）
- 遥测与追踪（Telemetry & Tracing）
- 多模型提供商支持（Bedrock, Anthropic, OpenAI 等）

---

## 架构设计理念

Strands Agent 采用了**事件驱动 + 递归循环**的架构模式，核心设计原则包括：

1. **模型驱动（Model-Driven）**: 将 LLM 作为推理引擎，由模型决定工具调用
2. **事件驱动（Event-Driven）**: Hook 系统提供松耦合的扩展点
3. **递归执行（Recursive Execution）**: 工具执行后递归调用事件循环，形成自然的推理链
4. **流式优先（Streaming-First）**: 底层全部基于 AsyncGenerator 实现流式处理
5. **可观测性（Observability）**: 内置 OpenTelemetry 追踪和指标收集

---

## 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│  agent("prompt") / agent.tool.xxx() / agent.stream_async()     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                       Agent (agent.py)                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  • messages: List[Message]        # 对话历史              │  │
│  │  • model: Model                   # LLM 提供商            │  │
│  │  • tool_registry: ToolRegistry    # 工具注册表            │  │
│  │  • hooks: HookRegistry            # 钩子系统              │  │
│  │  • state: AgentState              # 状态管理              │  │
│  │  • conversation_manager           # 会话管理              │  │
│  │  • session_manager                # 持久化管理            │  │
│  │  • _interrupt_state               # 中断状态              │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────▼────────────┐
                │  invoke_async()        │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │  stream_async()        │
                │   - 转换 prompt       │
                │   - 触发 hooks        │
                └────────────┬────────────┘
                             │
       ╔═════════════════════▼═════════════════════╗
       ║      Event Loop (event_loop.py)          ║
       ║  核心执行引擎 - 驱动整个 Agent 运行       ║
       ╚═════════════════════╤═════════════════════╝
                             │
         ┌───────────────────┴───────────────────┐
         │                                       │
   ┌─────▼──────┐                         ┌─────▼─────┐
   │ Model Call │                         │ Tool Call │
   └─────┬──────┘                         └─────┬─────┘
         │                                       │
   ┌─────▼──────────────────┐           ┌───────▼────────────────┐
   │ streaming.py           │           │ ToolExecutor           │
   │  - stream_messages()   │           │  - _stream()           │
   │  - process_stream()    │           │  - _stream_with_trace()│
   │                        │           │                        │
   │ ┌────────────────────┐ │           │ ┌────────────────────┐ │
   │ │ Model Provider     │ │           │ │ Python Tool        │ │
   │ │  - Bedrock         │ │           │ │ MCP Tool           │ │
   │ │  - OpenAI          │ │           │ │ Agent Tool         │ │
   │ │  - Anthropic       │ │           │ └────────────────────┘ │
   │ └────────────────────┘ │           └────────────────────────┘
   └────────────────────────┘

         ┌───────────────────────────────────────┐
         │        Hook System (hooks/)           │
         │  BeforeInvocationEvent                │
         │  AfterInvocationEvent                 │
         │  BeforeModelCallEvent                 │
         │  AfterModelCallEvent                  │
         │  BeforeToolCallEvent  ←── 可取消/中断 │
         │  AfterToolCallEvent   ←── 可修改结果   │
         │  MessageAddedEvent                    │
         └───────────────────────────────────────┘
```

---

## 核心执行流程

### 流程 1: Agent 初始化

```
agent.py:__init__() [213-366行]
│
├─► 1. 初始化模型 (BedrockModel 或自定义)
│   └─ model = BedrockModel() if not model else ...
│
├─► 2. 初始化基础属性
│   ├─ messages: List[Message] = []
│   ├─ system_prompt: str | List[SystemContentBlock]
│   └─ agent_id, name, description
│
├─► 3. 初始化工具系统
│   ├─ tool_registry = ToolRegistry()
│   ├─ tool_registry.process_tools(tools)  # 处理工具列表
│   ├─ tool_registry.initialize_tools()    # 发现并加载工具
│   └─ tool_watcher = ToolWatcher()        # 可选：监视工具文件变化
│
├─► 4. 初始化会话与状态
│   ├─ state = AgentState(initial_state)
│   ├─ conversation_manager = SlidingWindowConversationManager()
│   └─ session_manager (可选)
│
├─► 5. 初始化 Hook 系统
│   ├─ hooks = HookRegistry()
│   ├─ hooks.add_hook(hook_provider)
│   └─ hooks.invoke_callbacks(AgentInitializedEvent)
│
├─► 6. 初始化其他组件
│   ├─ tool_executor = ConcurrentToolExecutor()
│   ├─ callback_handler = PrintingCallbackHandler()
│   ├─ event_loop_metrics = EventLoopMetrics()
│   ├─ tracer = get_tracer()
│   └─ _interrupt_state = _InterruptState()
│
└─► 完成初始化
```

**关键文件**：
- `src/strands/agent/agent.py:213-366` - Agent 构造函数
- `src/strands/tools/registry.py:31-44` - ToolRegistry 初始化
- `src/strands/hooks/registry.py:154-156` - HookRegistry 初始化

---

### 流程 2: Agent 调用（核心执行流程）

```
用户调用: agent("你好")
│
├─► agent.__call__() [393-432行]
│   └─► 同步包装器，调用 run_async(invoke_async)
│
├─► agent.invoke_async() [434-474行]
│   ├─ 调用 stream_async()
│   └─ 等待所有事件完成，返回 AgentResult
│
└─► agent.stream_async() [597-683行] ⭐ 核心入口
    │
    ├─► 1. 处理中断状态
    │   └─ _interrupt_state.resume(prompt)
    │
    ├─► 2. 转换 prompt 为 Messages
    │   └─ _convert_prompt_to_messages(prompt) [782-826行]
    │       ├─ str → [{"role": "user", "content": [{"text": "..."}]}]
    │       ├─ List[ContentBlock] → Message
    │       └─ List[Message] → Messages
    │
    ├─► 3. 开启追踪 Span
    │   └─ trace_span = _start_agent_trace_span(messages)
    │
    ├─► 4. 触发 BeforeInvocationEvent
    │   └─ hooks.invoke_callbacks_async(BeforeInvocationEvent)
    │
    ├─► 5. 执行事件循环 ⭐⭐⭐
    │   └─► _run_loop(messages, invocation_state, structured_output_model)
    │       │
    │       ├─► 添加消息到历史
    │       │   └─ _append_message(message) [977-980行]
    │       │       └─ hooks.invoke_callbacks_async(MessageAddedEvent)
    │       │
    │       ├─► 初始化结构化输出上下文
    │       │   └─ StructuredOutputContext(structured_output_model)
    │       │
    │       └─► _execute_event_loop_cycle() [735-780行]
    │           └─► event_loop_cycle() ⭐⭐⭐ (event_loop.py:79-242)
    │
    ├─► 6. 应用会话管理策略
    │   └─ conversation_manager.apply_management(self)
    │
    ├─► 7. 触发 AfterInvocationEvent
    │   └─ hooks.invoke_callbacks_async(AfterInvocationEvent)
    │
    └─► 8. 返回 AgentResult
        └─ AgentResult(stop_reason, message, metrics, state)
```

**关键文件**：
- `src/strands/agent/agent.py:393-683` - Agent 调用链路
- `src/strands/event_loop/event_loop.py:79-242` - 核心事件循环

---

### 流程 3: Event Loop 执行（最核心）

```
event_loop_cycle() [event_loop.py:79-242]
│
├─► 1. 初始化循环状态
│   ├─ event_loop_cycle_id = uuid.uuid4()
│   ├─ cycle_trace = agent.event_loop_metrics.start_cycle()
│   └─ cycle_span = tracer.start_event_loop_cycle_span()
│
├─► 2. 检查是否需要模型调用
│   │
│   ├─► 情况 A: 中断状态激活
│   │   └─ 跳过模型调用，使用缓存的 tool_use_message
│   │
│   ├─► 情况 B: 最新消息包含 ToolUse
│   │   └─ 跳过模型调用，直接处理工具
│   │
│   └─► 情况 C: 需要模型调用
│       └─► _handle_model_execution() [283-418行]
│           │
│           ├─► 触发 BeforeModelCallEvent
│           │   └─ hooks.invoke_callbacks_async(BeforeModelCallEvent)
│           │
│           ├─► 调用模型（支持重试）
│           │   └─► stream_messages() [streaming.py:416-456]
│           │       ├─ _normalize_messages(messages)  # 清理空白文本
│           │       ├─ model.stream(messages, tool_specs, system_prompt)
│           │       └─► process_stream(chunks) [361-413行]
│           │           ├─ 处理 messageStart → 设置 role
│           │           ├─ 处理 contentBlockStart → 初始化 tool_use
│           │           ├─ 处理 contentBlockDelta → 累积文本/工具输入
│           │           ├─ 处理 contentBlockStop → 完成内容块
│           │           ├─ 处理 messageStop → 返回 stop_reason
│           │           └─ 处理 metadata → 提取使用指标
│           │
│           ├─► 触发 AfterModelCallEvent
│           │   └─ hooks.invoke_callbacks_async(AfterModelCallEvent)
│           │
│           ├─► 添加响应消息到历史
│           │   └─ agent.messages.append(message)
│           │
│           └─► 更新指标
│               └─ event_loop_metrics.update_usage(usage)
│
├─► 3. 根据 stop_reason 分支处理
│   │
│   ├─► 情况 A: stop_reason == "max_tokens"
│   │   └─ 抛出 MaxTokensReachedException
│   │
│   ├─► 情况 B: stop_reason == "tool_use" ⭐
│   │   └─► _handle_tool_execution() [420-533行]
│   │       │
│   │       ├─► 1. 验证并准备工具
│   │       │   └─ validate_and_prepare_tools(message, tool_uses, tool_results)
│   │       │
│   │       ├─► 2. 处理中断状态
│   │       │   └─ 从 interrupt_state.context 恢复 tool_results
│   │       │
│   │       ├─► 3. 执行工具 ⭐⭐
│   │       │   └─► tool_executor._execute()
│   │       │       └─ 见"流程 4: 工具执行"
│   │       │
│   │       ├─► 4. 检查中断
│   │       │   ├─ 如果有 interrupts:
│   │       │   │   ├─ _interrupt_state.activate()
│   │       │   │   └─ yield EventLoopStopEvent("interrupt", ...)
│   │       │   └─ 否则: _interrupt_state.deactivate()
│   │       │
│   │       ├─► 5. 添加工具结果消息
│   │       │   └─ agent.messages.append(tool_result_message)
│   │       │
│   │       ├─► 6. 检查是否继续循环
│   │       │   ├─ 如果 stop_event_loop 或 structured_output.stop_loop:
│   │       │   │   └─ yield EventLoopStopEvent(...)
│   │       │   └─ 否则:
│   │       │       └─► recurse_event_loop() [244-281行]
│   │       │           └─ 递归调用 event_loop_cycle()
│   │       │
│   │       └─► 结束工具执行
│   │
│   └─► 情况 C: stop_reason == "end_turn"
│       ├─ 检查是否需要强制结构化输出
│       │   └─ 如果 structured_output_context.is_enabled:
│       │       ├─ 设置 forced_mode
│       │       └─ recurse_event_loop()  # 强制调用工具
│       └─ yield EventLoopStopEvent(stop_reason, message, metrics, state)
│
└─► 4. 结束循环
    ├─ event_loop_metrics.end_cycle()
    └─ tracer.end_event_loop_cycle_span()
```

**关键文件**：
- `src/strands/event_loop/event_loop.py:79-533` - 事件循环核心
- `src/strands/event_loop/streaming.py:361-456` - 流式处理

---

### 流程 4: 工具执行

```
tool_executor._execute()
│
├─► ConcurrentToolExecutor._execute() [concurrent.py]
│   └─ 并发执行多个工具
│       └─► 对每个 tool_use:
│           └─► ToolExecutor._stream_with_trace() [_executor.py:221-274]
│
├─► _stream_with_trace() [221-274行]
│   ├─ 开启工具追踪 span
│   │   └─ tool_call_span = tracer.start_tool_call_span(tool_use)
│   │
│   ├─► 调用工具执行
│   │   └─► ToolExecutor._stream() [32-219行] ⭐
│   │
│   └─ 记录工具使用指标
│       └─ event_loop_metrics.add_tool_usage(tool_use, duration, trace)
│
└─► ToolExecutor._stream() [32-219行] ⭐⭐ 核心工具执行逻辑
    │
    ├─► 1. 查找工具
    │   ├─ tool_func = agent.tool_registry.registry.get(tool_name)
    │   └─ tool_spec = tool_func.tool_spec
    │
    ├─► 2. 触发 BeforeToolCallEvent
    │   └─► hooks.invoke_callbacks_async(BeforeToolCallEvent)
    │       ├─ 可以修改 tool_use
    │       ├─ 可以取消工具：cancel_tool = True
    │       └─ 可以触发中断：raise InterruptException
    │
    ├─► 3. 处理取消/中断
    │   ├─ 如果 cancel_tool:
    │   │   └─ yield ToolCancelEvent → return
    │   └─ 如果 interrupts:
    │       └─ yield ToolInterruptEvent → return
    │
    ├─► 4. 执行工具
    │   └─► selected_tool.stream(tool_use, invocation_state)
    │       ├─ 对于 Python 工具:
    │       │   └─ 执行 Python 函数
    │       ├─ 对于 MCP 工具:
    │       │   └─ 调用 MCP 服务器
    │       └─ yield ToolStreamEvent / ToolResultEvent
    │
    ├─► 5. 触发 AfterToolCallEvent
    │   └─► hooks.invoke_callbacks_async(AfterToolCallEvent)
    │       └─ 可以修改 result
    │
    ├─► 6. 返回工具结果
    │   └─ yield ToolResultEvent(result)
    │       └─ tool_results.append(result)
    │
    └─► 异常处理
        └─ 捕获异常 → 返回错误结果
            └─ result = {"status": "error", "content": [{"text": f"Error: {e}"}]}
```

**关键文件**：
- `src/strands/tools/executors/_executor.py:32-303` - 工具执行器基类
- `src/strands/tools/executors/concurrent.py` - 并发执行器

---

### 流程 5: Hook 系统

```
HookRegistry 工作流程
│
├─► 1. 注册 Hook
│   │
│   ├─► 方式 A: 注册单个回调
│   │   └─ registry.add_callback(EventType, callback)
│   │       └─ _registered_callbacks[EventType].append(callback)
│   │
│   └─► 方式 B: 注册 HookProvider
│       └─ registry.add_hook(hook_provider)
│           └─ hook_provider.register_hooks(registry)
│               └─ 批量注册多个回调
│
├─► 2. 触发事件
│   └─► hooks.invoke_callbacks_async(event) [202-246行]
│       │
│       ├─► 获取事件的回调列表
│       │   └─ get_callbacks_for(event) [309-335行]
│       │       ├─ 查找 _registered_callbacks[type(event)]
│       │       └─ 根据 event.should_reverse_callbacks 决定顺序
│       │
│       ├─► 遍历执行回调
│       │   └─ for callback in callbacks:
│       │       ├─ 如果是异步: await callback(event)
│       │       └─ 如果是同步: callback(event)
│       │
│       ├─► 收集中断
│       │   └─ 捕获 InterruptException
│       │       └─ interrupts[interrupt.name] = interrupt
│       │
│       └─► 返回
│           └─ return (event, list(interrupts.values()))
│
└─► 3. Hook 事件类型
    ├─ AgentInitializedEvent - Agent 初始化后
    ├─ BeforeInvocationEvent - Agent 调用前
    ├─ AfterInvocationEvent - Agent 调用后
    ├─ MessageAddedEvent - 消息添加到历史
    ├─ BeforeModelCallEvent - 模型调用前
    ├─ AfterModelCallEvent - 模型调用后
    ├─ BeforeToolCallEvent - 工具调用前（可修改/取消）
    └─ AfterToolCallEvent - 工具调用后（可修改结果）
```

**关键文件**：
- `src/strands/hooks/registry.py:143-336` - Hook 注册表
- `src/strands/hooks/events.py` - Hook 事件定义

---

### 流程 6: 会话管理

```
SessionManager & ConversationManager
│
├─► SessionManager 流程
│   │
│   ├─► 1. 通过 Hook 集成
│   │   └─ agent = Agent(session_manager=session_manager)
│   │       └─ hooks.add_hook(session_manager)
│   │
│   ├─► 2. 监听事件
│   │   ├─ MessageAddedEvent → 保存消息
│   │   ├─ AfterInvocationEvent → 同步 agent 状态
│   │   └─ 持久化到存储（文件/S3/自定义）
│   │
│   └─► 3. 恢复会话
│       └─ session_manager.load_session(agent_id)
│           ├─ 恢复 messages
│           ├─ 恢复 state
│           └─ 恢复 conversation_manager state
│
└─► ConversationManager 流程
    │
    ├─► 1. 应用管理策略
    │   └─ conversation_manager.apply_management(agent)
    │       │
    │       ├─ SlidingWindowConversationManager:
    │       │   └─ 保持最近 N 条消息
    │       │
    │       ├─ SummarizingConversationManager:
    │       │   └─ 总结旧消息，保留总结
    │       │
    │       └─ NullConversationManager:
    │           └─ 不做任何处理
    │
    └─► 2. 上下文溢出处理
        └─ conversation_manager.reduce_context(agent, exception)
            │
            ├─ 当 ContextWindowOverflowException 发生时调用
            │
            └─ 策略:
                ├─ 删除最老的 N 条消息
                ├─ 或总结历史对话
                └─ 递归重试 event_loop_cycle()
```

**关键文件**：
- `src/strands/session/session_manager.py` - 会话管理器
- `src/strands/agent/conversation_manager/` - 对话管理器

---

### 流程 7: 结构化输出

```
结构化输出执行流程
│
├─► 1. 初始化
│   └─ agent = Agent(structured_output_model=MyModel)
│       或
│       agent("prompt", structured_output_model=MyModel)
│
├─► 2. 注册结构化输出工具
│   └─ StructuredOutputContext(structured_output_model)
│       ├─ 创建内部工具: "__structured_output__"
│       └─ register_tool(tool_registry)
│
├─► 3. 模型调用
│   ├─ 正常模式: tool_choice = "auto"
│   └─ 强制模式: tool_choice = {"tool": {"name": "__structured_output__"}}
│
├─► 4. 工具执行
│   └─ 当模型调用 __structured_output__ 工具:
│       ├─ 提取工具输入
│       ├─ 验证并解析为 Pydantic 模型
│       └─ structured_output_context.extract_result(tool_uses)
│
├─► 5. 强制结构化输出
│   └─ 如果 stop_reason == "end_turn" 且未使用工具:
│       ├─ structured_output_context.set_forced_mode()
│       ├─ 添加提示消息: "You must format the previous response as structured output."
│       └─ recurse_event_loop()  # 强制下一轮调用工具
│
└─► 6. 返回结果
    └─ AgentResult.structured_output = parsed_model
```

**关键文件**：
- `src/strands/tools/structured_output/` - 结构化输出实现
- `src/strands/event_loop/event_loop.py:222-239` - 强制结构化输出逻辑

---

### 流程 8: 中断机制

```
Interrupt (人在循环中) 流程
│
├─► 1. 触发中断
│   └─ 在 BeforeToolCallEvent 回调中:
│       └─ raise InterruptException(Interrupt(
│           name="confirm_action",
│           data={"tool": tool_name, "params": params}
│       ))
│
├─► 2. 中断传播
│   └─► HookRegistry.invoke_callbacks_async()
│       ├─ 捕获 InterruptException
│       ├─ 收集到 interrupts 列表
│       └─ return (event, interrupts)
│
├─► 3. 事件循环处理中断
│   └─► _handle_tool_execution() [event_loop.py:420-533]
│       ├─ tool_events = tool_executor._execute(...)
│       ├─ 收集 ToolInterruptEvent
│       └─ if interrupts:
│           ├─ _interrupt_state.activate(context={
│           │   "tool_use_message": message,
│           │   "tool_results": tool_results
│           │ })
│           └─ yield EventLoopStopEvent("interrupt", ...)
│
├─► 4. 暂停执行
│   └─ Agent 返回 AgentResult(stop_reason="interrupt")
│       └─ 用户可以访问 result.interrupts
│
├─► 5. 恢复执行
│   └─ 用户确认后:
│       └─ agent("继续" 或其他输入)
│           ├─ _interrupt_state.resume(prompt)
│           ├─ 从保存的 context 恢复:
│           │   ├─ tool_use_message
│           │   └─ tool_results (已完成的工具)
│           └─ 继续执行剩余工具
│
└─► 6. 清理中断状态
    └─ _interrupt_state.deactivate()
```

**关键文件**：
- `src/strands/interrupt.py` - 中断状态管理
- `src/strands/hooks/registry.py:236-244` - 中断收集
- `src/strands/event_loop/event_loop.py:459-502` - 中断处理

---

## 设计模式与亮点

### 核心设计模式

#### 1. **递归驱动的推理链**
```python
# event_loop/event_loop.py:528-532
async def recurse_event_loop(...):
    """工具执行后递归调用事件循环，形成推理链"""
    events = event_loop_cycle(agent, invocation_state)
    async for event in events:
        yield event
```

这种设计使得 Agent 可以：
- 执行工具 → 获取结果 → 继续推理 → 再执行工具 → ...
- 形成自然的多轮对话和复杂任务执行

#### 2. **Hook 驱动的可扩展性**
```python
# 用户可以在任何阶段插入自定义逻辑
def my_hook(event: BeforeToolCallEvent):
    # 修改工具参数
    event.tool_use["input"]["max_results"] = 10
    # 或取消工具调用
    event.cancel_tool = True
    # 或触发人机交互
    raise InterruptException(Interrupt(...))
```

#### 3. **流式优先设计**
```python
# 所有核心函数都返回 AsyncGenerator
async def stream_async(self, prompt) -> AsyncIterator[Any]:
    async for event in self._run_loop(...):
        yield event  # 实时返回事件
```

#### 4. **异常驱动的上下文管理**
```python
# event_loop/event_loop.py:766-776
try:
    async for event in event_loop_cycle(...):
        yield event
except ContextWindowOverflowException as e:
    # 自动缩减上下文并重试
    conversation_manager.reduce_context(agent, e)
    async for event in self._execute_event_loop_cycle(...):
        yield event
```

### 设计亮点

1. **✅ 统一的事件模型**
   - 所有事件都继承自 `TypedEvent`
   - 支持回调处理和流式传递

2. **✅ 灵活的工具系统**
   - 支持多种工具定义方式（装饰器、模块、MCP）
   - 支持热重载

3. **✅ 强大的 Hook 机制**
   - 可以在任何阶段拦截和修改行为
   - 支持中断和恢复

4. **✅ 自动上下文管理**
   - 异常驱动的自动重试
   - 可插拔的历史管理策略

5. **✅ 完善的可观测性**
   - OpenTelemetry 集成
   - 详细的指标收集

6. **✅ 类型安全**
   - 全面的类型注解
   - 泛型支持结构化输出

---

## 核心数据流

```
1. 用户输入 (str/List[ContentBlock]/Messages)
   ↓
2. 转换为标准 Messages 格式
   ↓
3. 添加到 agent.messages 历史
   ↓
4. 调用模型 API (stream)
   ↓
5. 流式接收响应
   ├─ 文本块 → TextStreamEvent → 回调 → 用户
   ├─ 工具调用 → ToolUseStreamEvent → 回调 → 用户
   └─ 停止原因 → ModelStopReason
   ↓
6. 如果 stop_reason == "tool_use":
   ├─ 验证工具
   ├─ 执行工具 (并发/串行)
   ├─ 收集结果
   └─ 递归调用事件循环
   ↓
7. 如果 stop_reason == "end_turn":
   └─ 返回 AgentResult
   ↓
8. 应用 conversation_manager 策略
   └─ 管理历史消息长度
```

---

## 完整执行时序图

```
User          Agent         EventLoop      Model       ToolExecutor    Hooks
 │              │              │            │              │            │
 │─ "帮我搜索" ─►│              │            │              │            │
 │              │              │            │              │            │
 │              ├─ invoke_async()           │              │            │
 │              │              │            │              │            │
 │              ├───────────────── BeforeInvocationEvent ──────────────►│
 │              │              │            │              │            │
 │              ├─ stream_async()           │              │            │
 │              │              │            │              │            │
 │              ├─ _run_loop() ─►           │              │            │
 │              │              │            │              │            │
 │              │         event_loop_cycle()│              │            │
 │              │              │            │              │            │
 │              │              ├──────────── BeforeModelCallEvent ──────►│
 │              │              │            │              │            │
 │              │              ├─ stream() ─►              │            │
 │              │              │            │              │            │
 │              │              │◄─ chunks ──┤              │            │
 │◄─ 文本流 ─────┤◄─ events ────┤            │              │            │
 │              │              │            │              │            │
 │              │              ├──────────── AfterModelCallEvent ───────►│
 │              │              │            │              │            │
 │              │              │  [模型返回 tool_use]         │            │
 │              │              │            │              │            │
 │              │              ├─ _handle_tool_execution()  │            │
 │              │              │            │              │            │
 │              │              ├──────────────────────── BeforeToolCallEvent ─►│
 │              │              │            │              │            │
 │              │              │            │         _execute()        │
 │              │              │            │              │            │
 │              │              │            │         _stream()         │
 │              │              │            │              │            │
 │              │              │            │    ┌─ tool.stream() ─┐   │
 │              │              │            │    │   执行工具函数    │   │
 │              │              │            │    └─ yield result ──┘   │
 │              │              │            │              │            │
 │              │              ├──────────────────────── AfterToolCallEvent ──►│
 │              │              │            │              │            │
 │              │              │  [添加 tool_result 到 messages]          │
 │              │              │            │              │            │
 │              │              ├─ recurse_event_loop() ───┐│            │
 │              │              │            │              ││            │
 │              │         ┌────┴─ event_loop_cycle() ◄────┘│            │
 │              │         │    │            │              │            │
 │              │         │    ├─ stream() ─►              │            │
 │              │         │    │  [模型基于工具结果继续推理]  │            │
 │              │         │    │            │              │            │
 │              │         │    │◄─ "搜索结果是..." ─┤        │            │
 │◄─ 最终响应 ───┤◄─ events◄┴────┤            │              │            │
 │              │              │            │              │            │
 │              ├───────────────── AfterInvocationEvent ───────────────►│
 │              │              │            │              │            │
 │◄─ AgentResult┤              │            │              │            │
```

---

## 核心文件导航

### 按功能分类

| 文件 | 职责 | 核心函数 |
|------|------|----------|
| `agent/agent.py` | Agent 核心接口 | `__init__`, `__call__`, `invoke_async`, `stream_async` |
| `event_loop/event_loop.py` | 事件循环引擎 | `event_loop_cycle`, `recurse_event_loop` |
| `event_loop/streaming.py` | 流式处理 | `stream_messages`, `process_stream` |
| `tools/executors/_executor.py` | 工具执行抽象 | `_stream`, `_stream_with_trace` |
| `tools/registry.py` | 工具注册管理 | `process_tools`, `register_tool` |
| `hooks/registry.py` | Hook 系统 | `add_callback`, `invoke_callbacks_async` |
| `agent/state.py` | 状态管理 | `get`, `set`, `delete` |
| `agent/conversation_manager/` | 会话管理 | `apply_management`, `reduce_context` |
| `models/model.py` | 模型抽象接口 | `stream` |
| `interrupt.py` | 中断机制 | `_InterruptState` |

### 阅读顺序建议

如果你要深入研究源码，按这个顺序阅读最高效：

1. **入口理解**: `agent.py:393-683` - Agent 调用链
2. **核心引擎**: `event_loop/event_loop.py:79-242` - 事件循环
3. **模型交互**: `event_loop/streaming.py:416-456` - 流式处理
4. **工具执行**: `tools/executors/_executor.py:32-219` - 工具执行逻辑
5. **扩展机制**: `hooks/registry.py:202-246` - Hook 调用
6. **状态管理**: `agent/state.py` 和 `session/`

---

## 使用示例：理解工作流程

```python
from strands import Agent, tool

# 1. 定义工具
@tool
def search_web(query: str) -> str:
    """搜索互联网"""
    return f"搜索结果: {query}"

# 2. 创建 Agent（初始化流程）
agent = Agent(
    tools=[search_web],
    hooks=[MyLogger()]  # 注册 Hook
)

# 3. 调用 Agent（完整执行流程）
result = agent("谁发明了 Python？")

# 内部执行流程：
# ┌─ invoke_async()
# ├─ stream_async()
# ├─ BeforeInvocationEvent 触发
# ├─ event_loop_cycle() 开始
# │  ├─ BeforeModelCallEvent 触发
# │  ├─ 模型调用: "我需要搜索'Python 发明者'"
# │  ├─ AfterModelCallEvent 触发
# │  ├─ 模型返回 tool_use: search_web(query="Python 发明者")
# │  ├─ BeforeToolCallEvent 触发
# │  ├─ 执行 search_web() → "搜索结果: Python 发明者"
# │  ├─ AfterToolCallEvent 触发
# │  ├─ recurse_event_loop() 递归
# │  │  ├─ 模型再次调用: "根据搜索结果，Python 由 Guido van Rossum 发明"
# │  │  └─ stop_reason = "end_turn"
# │  └─ 返回 AgentResult
# ├─ AfterInvocationEvent 触发
# └─ 返回结果给用户
```

---

## 总结：核心设计精髓

作为 Strands Agent 的核心设计者视角，这个框架最精妙的地方在于：

### 1. **核心哲学：模型即推理引擎**
框架将 LLM 视为智能的决策中心，而不是简单的文本生成器。模型决定：
- 何时需要工具
- 调用哪些工具
- 如何组合工具结果继续推理

### 2. **递归驱动的执行模型**
```
用户输入 → 模型思考 → 工具执行 → 模型思考 → 工具执行 → ... → 最终答案
```
通过 `recurse_event_loop()` 实现的递归调用，优雅地实现了：
- 多轮推理
- 复杂任务分解
- 动态工作流调整

### 3. **事件驱动的可扩展性**
Hook 系统提供了 8 个关键拦截点，使得用户可以：
- 修改工具参数
- 取消工具调用
- 记录日志和指标
- 实现人机交互
- 自定义工具结果

### 4. **流式优先的用户体验**
所有核心函数都是 `AsyncGenerator`，确保：
- 实时响应反馈
- 低延迟交互
- 渐进式结果展示

### 5. **生产级的健壮性**
- 自动重试机制（模型限流）
- 上下文溢出自动处理
- 详细的追踪和指标
- 类型安全保证

---

## 贡献指南

如果你想为 Strands Agent 框架贡献代码，理解这些核心流程将帮助你：

1. **添加新的 Hook 事件**: 在 `hooks/events.py` 中定义新事件类
2. **实现自定义工具执行器**: 继承 `ToolExecutor` 并实现 `_execute()` 方法
3. **添加新的 ConversationManager 策略**: 继承 `ConversationManager` 并实现策略方法
4. **扩展模型提供商**: 实现 `Model` 接口

---

## 许可证

本文档遵循与 Strands Agents 项目相同的许可证（Apache License 2.0）。
