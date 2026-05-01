"""ChatModel 工厂。

统一管理 LLM 配置读取与 LangChain ChatModel 构建，避免业务层重复拼装参数。
"""

import asyncio
from typing import Optional
from itertools import cycle
from collections import deque

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_qwq import ChatQwen
from sqlmodel import Session

from app.db.models import LLMConfig
from app.services import llm_config_service

_llm_instance_semaphores: dict[int, asyncio.Semaphore] = {}
_llm_instance_concurrency: dict[int, int] = {}
_llm_instance_active: dict[int, int] = {}
_global_sem: Optional[asyncio.Semaphore] = None


def get_llm_status() -> dict:
    """获取 LLM 并发状态"""
    return {
        "instances": {
            config_id: {
                "limit": conc,
                "active": _llm_instance_active.get(config_id, 0),
            }
            for config_id, conc in _llm_instance_concurrency.items()
        },
        "global": {
            "limit": _get_global_sem()._value,
            "active": sum(_llm_instance_active.values()),
        }
    }


def _acquire_instance(llm_config_id: int) -> None:
    """增加实例活动计数"""
    _llm_instance_active[llm_config_id] = _llm_instance_active.get(llm_config_id, 0) + 1


def _release_instance(llm_config_id: int) -> None:
    """减少实例活动计数"""
    current = _llm_instance_active.get(llm_config_id, 0)
    _llm_instance_active[llm_config_id] = max(0, current - 1)


def _get_global_sem() -> asyncio.Semaphore:
    """获取全局信号量（自动计算所有实例并发之和）"""
    global _global_sem
    if _global_sem is None:
        total = sum(_llm_instance_concurrency.values()) or 30
        _global_sem = asyncio.Semaphore(max(1, total))
    return _global_sem


def _update_global_sem() -> None:
    """更新全局信号量（当实例配置变化时调用）"""
    global _global_sem
    total = sum(_llm_instance_concurrency.values()) or 30
    _global_sem = asyncio.Semaphore(max(1, total))


def _get_instance_semaphore(llm_config_id: int, concurrency: int = 4) -> asyncio.Semaphore:
    """获取或创建单个 LLM 实例的并发控制信号量"""
    if llm_config_id not in _llm_instance_semaphores:
        _llm_instance_semaphores[llm_config_id] = asyncio.Semaphore(max(1, concurrency))
        _llm_instance_concurrency[llm_config_id] = max(1, concurrency)
        _llm_instance_active[llm_config_id] = 0
        _update_global_sem()
    return _llm_instance_semaphores[llm_config_id]


def set_llm_instance_concurrency(llm_config_id: int, concurrency: int) -> None:
    """动态调整单个 LLM 实例的并发限制"""
    _llm_instance_semaphores[llm_config_id] = asyncio.Semaphore(max(1, concurrency))
    _llm_instance_concurrency[llm_config_id] = max(1, concurrency)
    _update_global_sem()


def set_llm_global_concurrency(limit: int) -> None:
    """设置全局并发限制（手动覆盖）"""
    global _global_sem
    _global_sem = asyncio.Semaphore(max(1, limit))


def _get_instance_concurrency(session: Session, llm_config_id: int) -> int:
    """获取 LLM 配置的并发限制（从 endpoints 累加）"""
    cfg = llm_config_service.get_llm_config(session, llm_config_id)
    if not cfg:
        return 4
    if cfg.endpoints:
        total = sum(ep.get("concurrency", 4) for ep in cfg.endpoints)
        return max(1, total)
    return getattr(cfg, "concurrency", 4)


def _sanitize_common_generation_kwargs(
    *,
    temperature: Optional[float],
    max_tokens: Optional[int],
    timeout: Optional[float],
) -> dict:
    normalized_max_tokens = None if max_tokens == -1 else max_tokens
    kwargs: dict = {}
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if normalized_max_tokens is not None:
        kwargs["max_tokens"] = int(normalized_max_tokens)
    if timeout is not None:
        kwargs["timeout"] = float(timeout)
    return kwargs


def _build_openai_family_transport_kwargs(transport: dict) -> dict:
    kwargs: dict = {}
    if transport["request_base"]:
        # LangChain 这里统一接收 `base_url`。
        # 之前传成 `openai_api_base` 会导致 `ChatQwen` 忽略自定义网关，
        # 旧的 openai_compatible 配置就会错误落回默认供应商端点。
        kwargs["base_url"] = transport["request_base"]
    if transport["default_headers"]:
        kwargs["default_headers"] = transport["default_headers"]
    if transport["api_protocol"] != "auto":
        kwargs["use_responses_api"] = transport["api_protocol"] == "responses"
    elif transport["use_responses_api"]:
        kwargs["use_responses_api"] = True
    return kwargs


def build_chat_model_from_payload(
    *,
    provider: str,
    model_name: str,
    api_key: str,
    api_base: str | None = None,
    base_url: str | None = None,
    api_protocol: str | None = None,
    custom_request_path: str | None = None,
    user_agent: str | None = None,
    endpoints: list[dict] | None = None,
    llm_config_id: int | None = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    thinking_enabled: Optional[bool] = None,
):
    if not api_key:
        raise ValueError("未提供 API Key")

    # 如果配置了端点池，优先从池中获取URL
    resolved_api_base = api_base
    if endpoints and llm_config_id:
        pool = llm_config_service.get_endpoint_pool(llm_config_id, endpoints)
        if pool:
            endpoint_url = pool.get_endpoint()
            if endpoint_url:
                resolved_api_base = endpoint_url
    elif endpoints and endpoints[0].get("url"):
        resolved_api_base = endpoints[0].get("url")

    transport = llm_config_service.resolve_transport_settings(
        provider=provider,
        api_base=resolved_api_base,
        base_url=base_url,
        api_protocol=api_protocol,
        custom_request_path=custom_request_path,
        user_agent=user_agent,
    )
    provider_name = transport["provider"]
    common_kwargs = _sanitize_common_generation_kwargs(
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    if provider_name in {"openai_compatible", "openai"}:
        model_kwargs = {
            "model": model_name,
            "api_key": api_key,
            **_build_openai_family_transport_kwargs(transport),
            **common_kwargs,
        }
        if thinking_enabled is not None:
            model_kwargs["extra_body"] = {"enable_thinking": thinking_enabled}

        # `responses` 模式下统一走 `ChatOpenAI`。
        # 原先 openai_compatible 仍走 `ChatQwen`，会在流式 continuation 时构造出
        # 不满足 openai-python Responses API 要求的 payload，触发
        # “Missing required arguments; Expected either ('messages' and 'model') ...”。
        if transport["use_responses_api"]:
            return ChatOpenAI(**model_kwargs)

        if provider_name == "openai_compatible":
            return ChatQwen(**model_kwargs)
        return ChatOpenAI(**model_kwargs)

    if provider_name == "anthropic":
        model_kwargs = {
            "model": model_name,
            "api_key": api_key,
            **common_kwargs,
        }
        if thinking_enabled is True:
            model_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 2048}
        return ChatAnthropic(**model_kwargs)

    if provider_name == "google":
        model_kwargs = {
            "model": model_name,
            "api_key": api_key,
        }
        if thinking_enabled is not None:
            model_kwargs["include_thoughts"] = thinking_enabled
        if common_kwargs.get("max_tokens") is not None:
            model_kwargs["max_output_tokens"] = common_kwargs["max_tokens"]
        if common_kwargs.get("temperature") is not None:
            model_kwargs["temperature"] = common_kwargs["temperature"]
        if common_kwargs.get("timeout") is not None:
            model_kwargs["timeout"] = common_kwargs["timeout"]
        return ChatGoogleGenerativeAI(**model_kwargs)

    raise ValueError(f"不支持的 LLM 提供商: {provider}")


def _get_llm_config(session: Session, llm_config_id: int) -> LLMConfig:
    cfg = llm_config_service.get_llm_config(session, llm_config_id)
    if not cfg:
        raise ValueError(f"LLM 配置不存在，ID: {llm_config_id}")
    if not cfg.api_key:
        raise ValueError(f"未找到 LLM 配置 {cfg.display_name or cfg.model_name} 的 API 密钥")
    return cfg


def _get_instance_concurrency(session: Session, llm_config_id: int) -> int:
    """获取 LLM 配置的并发限制"""
    cfg = _get_llm_config(session, llm_config_id)
    if cfg.endpoints:
        total = sum(ep.get("concurrency", 4) for ep in cfg.endpoints)
        return max(1, total)
    return getattr(cfg, "concurrency", 4)


def build_chat_model(
    session: Session,
    llm_config_id: int,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    thinking_enabled: Optional[bool] = None,
):
    cfg = _get_llm_config(session, llm_config_id)
    return build_chat_model_from_payload(
        provider=cfg.provider,
        model_name=cfg.model_name,
        api_key=cfg.api_key,
        api_base=cfg.api_base,
        base_url=cfg.base_url,
        api_protocol=getattr(cfg, "api_protocol", None),
        custom_request_path=getattr(cfg, "custom_request_path", None),
        user_agent=getattr(cfg, "user_agent", None),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        thinking_enabled=thinking_enabled,
    )
