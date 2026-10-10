import os
import json
import time
import requests
from dotenv import load_dotenv

from config.config import CURRENT_LLM_CONFIG

load_dotenv()


class GLMClient:
    """
    统一 LLM 客户端
    支持：
    - 远程 API 模型
    - 本地 OpenAI 兼容模型
    """

    def __init__(self, api_key=None):
        self.provider = CURRENT_LLM_CONFIG
        self.base_url = self.provider["base_url"]
        self.model = self.provider["model_name"]

        self.supports_thinking = self.provider.get("supports_thinking", False)
        self.supports_json_mode = self.provider.get("supports_json_mode", False)
        self.supports_system = self.provider.get("supports_system", True)
        self.fixed_temperature = self.provider.get("fixed_temperature", False)
        self.disable_thinking_by_default = self.provider.get("disable_thinking_by_default", False)
        self.default_temperature = self.provider.get("temperature", 0.1)
        self.default_max_tokens = self.provider.get("max_tokens", 512)
        self.is_local = self.provider.get("is_local", False)

        api_key_env = self.provider["api_key_env"]
        self.api_key = api_key or os.getenv(api_key_env)

        # 远程模型：必须有 key
        if not self.api_key and not self.is_local:
            raise ValueError(f"未找到 {api_key_env}，请先在 .env 或环境变量中配置。")

        # 本地模型：没有 key 也允许，给个占位值
        if not self.api_key and self.is_local:
            self.api_key = "local-test-key"

    def _normalize_messages(self, messages):
        """
        对不支持 system 的模型，把 system 合并进第一个 user。
        """
        if self.supports_system:
            return messages

        system_parts = []
        other_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                if content:
                    system_parts.append(content)
            elif role in ("user", "assistant"):
                other_messages.append({
                    "role": role,
                    "content": content
                })

        if system_parts:
            system_text = "\n".join(system_parts).strip()

            if other_messages and other_messages[0]["role"] == "user":
                first_content = other_messages[0].get("content", "")
                other_messages[0]["content"] = f"{system_text}\n\n{first_content}".strip()
            else:
                other_messages.insert(0, {
                    "role": "user",
                    "content": system_text
                })

        return other_messages

    def _apply_provider_specific_options(self, payload, use_thinking):
        """
        针对不同 provider 注入额外参数
        """
        base_url_lower = self.base_url.lower()
        model_lower = self.model.lower()

        # 智谱 thinking
        if "bigmodel.cn" in base_url_lower:
            if self.supports_thinking and use_thinking is True:
                payload["thinking"] = {"type": "enabled"}

        # Kimi thinking
        elif "moonshot.cn" in base_url_lower or "kimi" in model_lower:
            if self.supports_thinking:
                if use_thinking is True:
                    payload["thinking"] = {"type": "enabled"}
                elif self.disable_thinking_by_default:
                    payload["thinking"] = {"type": "disabled"}

        return payload

    def chat(self, messages, temperature=None, max_tokens=None, response_format=None, use_thinking=None):
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        normalized_messages = self._normalize_messages(messages)
        final_temperature = self.default_temperature if temperature is None else temperature

        payload = {
            "model": self.model,
            "messages": normalized_messages,
            "max_tokens": self.default_max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }

        # 固定温度模型（如 kimi）
        if self.fixed_temperature:
            payload["temperature"] = self.default_temperature
        else:
            payload["temperature"] = final_temperature

        payload = self._apply_provider_specific_options(payload, use_thinking)

        # 仅支持 json mode 的模型才传
        if response_format is not None and self.supports_json_mode:
            payload["response_format"] = response_format

        resp = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=180,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"API 调用失败: {resp.status_code}\n{resp.text}")

        data = resp.json()

        if "choices" not in data or not data["choices"]:
            raise RuntimeError(
                f"返回格式异常，未找到 choices：\n{json.dumps(data, ensure_ascii=False, indent=2)}"
            )

        return data

    def _extract_json_from_text(self, text: str):
        """
        从完整文本中提取 JSON
        """
        if not text:
            return None

        text = text.strip()
        if not text:
            return None

        # 情况1：整个文本就是 JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 情况2：文本中嵌了一个 JSON 对象
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        return None

    def _repair_truncated_json(self, text: str):
        """
        尝试修复被截断的 JSON。
        适用于 content 已经明显是 JSON 开头，但结尾被截断的情况。
        """
        if not text:
            return None

        text = text.strip()
        if not text.startswith("{"):
            return None

        candidates = [text]

        # 截到最后一个相对安全的位置
        cut_positions = []
        for token in [',\n    {', ',\n    "', ',\n  "', ',\n', ',']:
            pos = text.rfind(token)
            if pos != -1:
                cut_positions.append(pos)

        if cut_positions:
            candidates.append(text[:max(cut_positions)])

        for candidate in candidates:
            candidate = candidate.rstrip()

            fixed_versions = [
                candidate + '\n    }\n  ]\n}',
                candidate + '\n  ]\n}',
                candidate + '\n}',
                candidate + ']}',
            ]

            for fixed in fixed_versions:
                try:
                    return json.loads(fixed)
                except Exception:
                    continue

        return None

    def repair_to_json(self, original_text, reasoning_text, max_tokens=512):
        """
        当模型只给 reasoning_content、不直接给 JSON 时，
        再发一次短请求，把分析结果压缩成严格 JSON。
        """
        messages = [
            {
                "role": "user",
                "content": (
                    "请根据下面的分析结果，直接输出严格 JSON，不要解释。\n\n"
                    f"原句：{original_text}\n\n"
                    f"分析内容：{reasoning_text}\n\n"
                    "输出格式：\n"
                    "{\n"
                    '  "text": "原句",\n'
                    '  "entities": [\n'
                    "    {\n"
                    '      "entity": "实体文本",\n'
                    '      "entity_type": "实体类别",\n'
                    '      "start_idx": 0,\n'
                    '      "end_idx": 2\n'
                    "    }\n"
                    "  ]\n"
                    "}\n"
                )
            }
        ]

        result = self.chat(
            messages=messages,
            max_tokens=max_tokens,
            use_thinking=False
        )

        message = result["choices"][0].get("message", {})
        content = message.get("content", "") or ""

        parsed = self._extract_json_from_text(content)
        if parsed is not None:
            return parsed, result

        repaired = self._repair_truncated_json(content)
        if repaired is not None:
            return repaired, result

        return None, result

    def chat_json(
        self,
        messages,
        temperature=0.0,
        max_tokens=512,
        max_retries=1,
        use_thinking=False,
    ):
        """
        发送请求并尽量返回合法 JSON
        """
        last_result = None

        for attempt in range(max_retries + 1):
            result = self.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                use_thinking=use_thinking,
            )
            last_result = result

            message = result["choices"][0].get("message", {})
            content = message.get("content", "") or ""
            reasoning_content = message.get("reasoning_content", "") or ""

            # 1. 正常 JSON
            parsed = self._extract_json_from_text(content)
            if parsed is not None:
                return parsed, result

            # 2. 截断 JSON 修复
            repaired = self._repair_truncated_json(content)
            if repaired is not None:
                return repaired, result

            # 3. reasoning_content 本身就是 JSON
            parsed = self._extract_json_from_text(reasoning_content)
            if parsed is not None:
                return parsed, result

            # 4. 若只返回推理内容，则尝试二次压缩
            if not content and reasoning_content and messages:
                original_text = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        original_text = msg.get("content", "")
                        break

                repaired_json, repaired_result = self.repair_to_json(
                    original_text=original_text,
                    reasoning_text=reasoning_content,
                    max_tokens=512
                )
                if repaired_json is not None:
                    return repaired_json, repaired_result

            if attempt < max_retries:
                time.sleep(1.0)

        message = last_result["choices"][0].get("message", {}) if last_result else {}
        raise RuntimeError(
            "模型未返回合法 JSON。\n\n"
            f"content:\n{message.get('content', '')}\n\n"
            f"reasoning_content:\n{message.get('reasoning_content', '')}\n\n"
            f"完整返回:\n{json.dumps(last_result, ensure_ascii=False, indent=2) if last_result else 'None'}"
        )
