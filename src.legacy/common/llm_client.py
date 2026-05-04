"""MiniMax LLM 调用封装。"""
import json
import logging
from typing import Any

from openai import OpenAI, APIError

from .config import get_config

logger = logging.getLogger(__name__)


class LLMClient:
    """MiniMax LLM 调用客户端。"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        cfg = get_config()
        self.client = OpenAI(
            api_key=api_key or cfg.minimax_api_key,
            base_url=base_url or cfg.minimax_base_url,
            timeout=30.0,
        )
        self.model = model or cfg.minimax_model

    def text(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
    ) -> str:
        """发送对话请求，返回原始文本（不解析 JSON）。"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"LLM 调用失败: {e}")
            return ""

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        response_format: dict | None = None,
    ) -> dict[str, Any]:
        """
        发送对话请求，返回解析后的 JSON dict。

        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": ...}]
            temperature: 生成温度
            response_format: {"type": "json_object"} 则强制 JSON 输出

        Returns:
            解析后的 dict（由 LLM 返回的 JSON 内容）
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content
            if raw is None:
                return {}
            # 去除 markdown 代码块包裹
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                raw = raw.lstrip("json").lstrip("\n").rstrip("```").strip()
            # 尝试直接解析
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
            # 模型可能在 JSON 前后加了自然语言描述
            # 从第一个 { 开始，逐步扩展字符串直到能完整解析
            json_start = raw.find('{')
            if json_start == -1:
                return {}
            candidate = raw[json_start:]
            # 从短到长尝试，找到第一个能完整解析的
            for end_offset in range(20, len(candidate) + 1):
                try:
                    result = json.loads(candidate[:end_offset])
                    if isinstance(result, dict) and "type" in result:
                        return result
                except Exception:
                    continue
            return {}
        except APIError as e:
            logger.warning(f"LLM API 错误: {e}，返回空 dict")
            return {}
        except json.JSONDecodeError:
            logger.warning("LLM 返回内容无法解析为 JSON，返回空 dict")
            return {}

    def batch_chat(
        self,
        batch: list[dict[str, Any]],
        system_prompt: str,
        temperature: float = 0.1,
    ) -> list[dict[str, Any]]:
        """
        批量 chat。每条输入需要包含 "user" 字段。

        Args:
            batch: [{"user": "问题1"}, {"user": "问题2"}, ...]
            system_prompt: 系统提示词
            temperature: 生成温度

        Returns:
            与 batch 同序的 dict 列表
        """
        results = []
        for item in batch:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": item["user"]}]
            results.append(self.chat(messages, temperature=temperature))
        return results
