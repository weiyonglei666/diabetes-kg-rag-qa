import json
from utils.glm_client import GLMClient
from utils.prompts import NER_SYSTEM_PROMPT
from data.postprocess import validate_ner_result


def extract_entities(text: str):
    client = GLMClient()

    messages = [
        {"role": "system", "content": NER_SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ]

    ner_result, raw_result = client.chat_json(
        messages,
        temperature=0.0,
        max_tokens=2048,
        max_retries=0,
        use_thinking=False
    )

    ner_result = validate_ner_result(text, ner_result)
    return ner_result, raw_result


def main():
    text = "随着社会经济的发展，糖尿病患病率逐渐增加，已成为严重的世界性问题。糖尿病特别是其慢性并发症影响患者的生活质量，甚至威胁患者的生命，给社会、家庭以及患者带来沉重的经济负担。中国的流行病学调查显示，中国20岁以上人群2型糖尿病患病率达9.7%；且近2/3的患者HbA1c得不到有效控制(HbA1c≤7%)；60.7%的患者因未被诊断而无法及早进行有效的治疗和指导。"

    ner_result, raw_result = extract_entities(text)

    print("=" * 60)
    print("解析并校验后的实体识别结果：")
    print(json.dumps(ner_result, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("原始模型返回：")
    print(json.dumps(raw_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
