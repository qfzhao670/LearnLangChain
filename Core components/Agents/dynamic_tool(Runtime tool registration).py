from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest


# -----------------------
# 1️⃣ 静态工具（原本就有）
# -----------------------
@tool
def get_weather(location: str) -> str:
    """Get weather for a given city."""
    return f"The weather in {location} is sunny."


# -----------------------
# 2️⃣ 动态工具（不会写在 create_agent 里）--->>>我猜意义在于：有些工具可能是别人提供的，我们无法一开始就写到tools里
# -----------------------
@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 20.0) -> str:
    """Calculate the tip amount for a bill."""
    tip = bill_amount * (tip_percentage / 100)
    return f"Tip: ${tip:.2f}, Total: ${bill_amount + tip:.2f}"


# 用户输入
#   ↓
# 模型决定是否调用工具
#   ↓
# 如果调用 → 执行工具
#   ↓
# 把结果喂回模型
#   ↓
# 模型生成最终答案

# -----------------------
# 3️⃣ Middleware
# -----------------------
class DynamicToolMiddleware(AgentMiddleware):

    def wrap_model_call(self, request: ModelRequest, handler):
        print("\n[Middleware] ===== 模型调用前 =====")
        print("[Middleware] 原始工具:", [t.name for t in request.tools])

        # 注入动态工具
        updated = request.override(
            tools=[*request.tools, calculate_tip]
        )

        print("[Middleware] 注入工具: calculate_tip")
        print("[Middleware] 当前工具:", [t.name for t in updated.tools])

        return handler(updated)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        tool_name = request.tool_call["name"]

        print("\n[Middleware] ===== 工具调用阶段 =====")
        print(f"[Middleware] 模型选择调用工具: {tool_name}")

        if tool_name == "calculate_tip":
            print("[Middleware] 正在执行动态工具: calculate_tip")

            return handler(
                request.override(tool=calculate_tip)
            )

        return handler(request)


# -----------------------
# 4️⃣ 创建 Agent
# -----------------------
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
    model=model,  # 使用初始化的模型
    tools=[get_weather],  # ❗只有静态工具
    middleware=[DynamicToolMiddleware()],
)


# -----------------------
# 5️⃣ 测试 1：调用动态工具
# -----------------------
print("\n====== 测试1：Tip 计算 ======")

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Calculate a 20% tip on $85"}
    ]
})

print("\n[最终结果]")
print(result["messages"][-1].content)


# -----------------------
# 6️⃣ 测试 2：调用静态工具
# -----------------------
print("\n====== 测试2：天气查询 ======")

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What's the weather in Paris?"}
    ]
})

print("\n[最终结果]")
print(result["messages"][-1].content)