from typing import Callable
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool


# ------------------------
# 1️⃣ 定义工具
# ------------------------

@tool
def public_search(query: str) -> str:
    """Perform a public search for a given query."""
    return f"[PUBLIC] 搜索结果: {query}"

@tool
def private_search(query: str) -> str:
    """Perform a private search for authenticated users."""
    return f"[PRIVATE] 私密搜索结果: {query}"

@tool
def advanced_search(query: str) -> str:
    """Perform an advanced search, available after multiple messages."""
    return f"[ADVANCED] 高级搜索结果: {query}"


# ------------------------
# 2️⃣ Middleware：动态工具控制
# ------------------------
# 用户输入 
#     ↓
# [中间件执行] ← 第一次
#     ↓
# 模型调用（决定用 public_search）
#     ↓
# 执行 public_search 工具
#     ↓
# [中间件执行] ← 第二次
#     ↓
# 模型调用（基于工具结果生成回答）
#     ↓
# 返回最终回答
@wrap_model_call    # wrap_model_call在调用大模型前就会执行state_based_tools这个函数
def state_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:

    state = request.state
    additional_kwargs = state['messages'][0].additional_kwargs

    is_authenticated = additional_kwargs.get("authenticated", False)

    print("\n=== Middleware执行 ===")
    print("是否登录:", is_authenticated)
    print("原始工具:", [t.name for t in request.tools])

    # ❗未登录：只能用 public
    if not is_authenticated:
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)

    else:
        tools = request.tools
        request = request.override(tools=tools)

    print("最终工具:", [t.name for t in request.tools])

    return handler(request)


# ------------------------
# 3️⃣ 创建 Agent
# ------------------------

model = init_chat_model(
    model="qwen3.5-plus",
    model_provider="openai",
    api_key="sk-8710483a982e426aa08765f872601588",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.1,
    extra_body={
        "enable_thinking": False   # 👈 关键：关闭 thinking mode
    }
)

agent = create_agent(
    model=model,
    tools=[public_search, private_search, advanced_search],
    middleware=[state_based_tools]
)


# ------------------------
# 4️⃣ 模拟不同 State 调用
# ------------------------

print("\n====== 未登录测试 ======")
resp1 = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我搜索LangChain", "authenticated": False}]},
)
print("tool-------------------------------------------------------")
print(resp1["messages"][-2].name)  # 打印调用的工具名称
print("response-------------------------------------------------------")
print(resp1["messages"][-1].content)

print("\n====== 已登录 ======")
resp2 = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我做高级分析", "authenticated": True}]},
)
print("tool-------------------------------------------------------")
print(resp2["messages"][-2].name)  # 打印调用的工具名称
print("response-------------------------------------------------------")
print(resp2["messages"][-1].content)