from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool


# -----------------------
# 1️⃣ 工具1：搜索商品
# -----------------------
@tool
def search_products(query: str) -> str:
    """Search for products based on a query."""
    print(f"\n[Tool] search_products 被调用，query={query}")

    # 模拟返回结果（真实场景可以接 API）
    return "Top products: WH-1000XM5, AirPods Pro 2, Bose QC Ultra"


# -----------------------
# 2️⃣ 工具2：查库存
# -----------------------
@tool
def check_inventory(product_id: str) -> str:
    """Check inventory for a specific product. """
    print(f"\n[Tool] check_inventory 被调用，product_id={product_id}")

    # 模拟库存
    if product_id == "WH-1000XM5":
        return "Product WH-1000XM5: 10 units in stock"
    return f"Product {product_id}: Out of stock"


# -----------------------
# 3️⃣ 模型（关闭thinking方便观察）
# -----------------------
model = init_chat_model(
    model="qwen3.5-plus",
    model_provider="openai",
    api_key="sk-8710483a982e426aa08765f872601588",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
    extra_body={
        "enable_thinking": False
    }
)


# -----------------------
# 4️⃣ Agent
# -----------------------
agent = create_agent(
    model=model,
    tools=[search_products, check_inventory],
)


# -----------------------
# 5️⃣ 执行任务（ReAct循环）
# -----------------------
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Find the most popular wireless headphones right now and check if they're in stock"
        }
    ]
})


# -----------------------
# 6️⃣ 打印完整过程
# -----------------------
print("\n====== 完整消息流 ======")
for msg in result["messages"]:
    print(f"\n[{type(msg).__name__}]")
    print(msg)