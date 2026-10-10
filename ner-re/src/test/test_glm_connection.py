import json
from utils.glm_client import GLMClient
from config.config import LLM_PROVIDER, CURRENT_LLM_CONFIG


def main():
    client = GLMClient()

    messages = [
        {
            "role": "system",
            "content": "你是一个严谨的中文医学信息抽取助手。"
        },
        {
            "role": "user",
            "content": "请只回复：连接成功。不要输出其他内容。"
        }
    ]

    result = client.chat(
        messages,
        temperature=0.0,
        max_tokens=32,
        use_thinking=False
    )

    print("=" * 60)
    print(f"当前 provider: {LLM_PROVIDER}")
    print(f"base_url: {CURRENT_LLM_CONFIG['base_url']}")
    print(f"model_name: {CURRENT_LLM_CONFIG['model_name']}")
    print("=" * 60)
    print("完整返回：")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("模型回复：")
    print(result["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
