from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

# ========== 1. 定义两个模型 ==========
basic_model = init_chat_model(
    model="qwen3.5-35b-a3b",
    model_provider="openai",
    api_key="sk-8710483a982e426aa08765f872601588",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.1,
    extra_body={"enable_thinking": False}
)

advanced_model = init_chat_model(
    model="qwen3.5-plus",
    model_provider="openai",
    api_key="sk-8710483a982e426aa08765f872601588",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.1,
    extra_body={"enable_thinking": False}
)

# agent.invoke()
#    ↓
# dynamic_model_selection  👈（中间件）
#    ↓
# 真正调用模型

# ========== 2. 动态模型选择 ==========
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state["messages"])

    if message_count > 5:
        model = advanced_model
        print(f"\n🔥 使用高级模型（thinking OFF），消息数: {message_count}")
    else:
        model = basic_model
        print(f"\n⚡ 使用基础模型（thinking OFF），消息数: {message_count}")

    # 👉 “我把 request 里的 model 换掉，然后交给系统继续执行”
    return handler(request.override(model=model))


# ========== 3. 创建 Agent ==========
agent = create_agent(
    model=basic_model,
    middleware=[dynamic_model_selection]
)

# ========== 4. 模拟对话 ==========
messages = []

def chat(user_input):
    messages.append({"role": "user", "content": user_input})

    response = agent.invoke({
        "messages": messages
    })

    output = response["messages"][-1].content
    messages.append({"role": "assistant", "content": output})

    print(f"🤖: {output}")


# ========== 5. 测试 ==========
chat("你好")
chat("给我讲个笑话")
chat("再讲一个")
chat("继续")            # 👈 这里应该切换到 advanced_model
chat("再来一个")
chat("解释一下量子力学")  