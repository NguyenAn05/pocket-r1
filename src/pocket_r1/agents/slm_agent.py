"""SLM Agent wrapper supporting OpenAI-compatible endpoints (vLLM/Ollama/API) and Mock mode."""

from __future__ import annotations

import os

from pocket_r1.agents.base import AgentResult, BaseAgent
from pocket_r1.env.base import Observation
from pocket_r1.prompts.parser import parse_poker_cot
from pocket_r1.prompts.state_formatter import format_observation
from pocket_r1.prompts.templates import build_chat_messages
from pocket_r1.verifiers.card_math import (
    compute_pot_odds,
    count_draws_and_outs,
    estimate_equity,
    evaluate_hand_category,
)


class SLMAgent(BaseAgent):
    """Calls an OpenAI-compatible chat endpoint (vLLM, Ollama, OpenAI, etc.)."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 512,
        few_shot: bool = False,
        enable_thinking: bool = False,
        timeout: float = 120.0,
    ) -> None:
        from openai import OpenAI

        self.name = model_name
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.few_shot = few_shot
        self.enable_thinking = enable_thinking
        resolved_key = api_key or os.environ.get(api_key_env, "EMPTY")
        self.client = OpenAI(base_url=base_url, api_key=resolved_key, timeout=timeout)

    def act(self, obs: Observation) -> AgentResult:
        prompt_text = format_observation(obs)
        messages = build_chat_messages(obs, few_shot=self.few_shot)
        kwargs: dict = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if not self.enable_thinking and "qwen3" in self.model_name.lower():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        completion = self.client.chat.completions.create(**kwargs)
        raw_text = completion.choices[0].message.content or ""
        parsed = parse_poker_cot(raw_text, legal_actions=obs.legal_actions)
        return AgentResult(
            action=parsed.resolved_action,
            raw_response=raw_text,
            prompt_text=prompt_text,
            parsed=parsed,
            metadata={"model": self.model_name},
        )


class MockSLMAgent(BaseAgent):
    """Offline mock SLM agent that emits valid 5-step Poker CoT for pipeline verification."""

    def __init__(self, inner_bot: BaseAgent | None = None, seed: int = 42) -> None:
        from pocket_r1.agents.rule_bots import create_bot

        self.name = "mock_slm"
        self.inner_bot = inner_bot or create_bot("tag", seed=seed)

    def act(self, obs: Observation) -> AgentResult:
        prompt_text = format_observation(obs)
        bot_res = self.inner_bot.act(obs)
        action = bot_res.action

        hand_cat = evaluate_hand_category(obs.hero_cards, obs.board)
        draws = count_draws_and_outs(obs.hero_cards, obs.board)
        pot_odds = compute_pot_odds(obs.pot_bb, obs.to_call_bb)
        equity = estimate_equity(obs.hero_cards, obs.board, n_samples=80, seed=obs.seed)
        board_str = " ".join(obs.board) if obs.board else "None"
        hole_str = " ".join(obs.hero_cards)
        cmp_op = ">" if equity >= pot_odds else "<"

        outs_text = (
            f" với {draws['total_draw_outs']} outs mua bài"
            if draws["total_draw_outs"] > 0
            else ""
        )
        raw_text = (
            f"<step>1. Tình huống: Vị trí {obs.position}, Hero stack {obs.hero_stack_bb:g}BB, "
            f"bài tẩy {hole_str}, mặt bàn {board_str}, Current Pot {obs.pot_bb:g}BB, "
            f"To Call {obs.to_call_bb:g}BB.</step>\n"
            f"<step>2. Sức mạnh bài: Tay bài hiện tại là {hand_cat}{outs_text}.</step>\n"
            f"<step>3. Toán học: Pot Odds = {obs.to_call_bb:g} / ({obs.pot_bb:g} + {obs.to_call_bb:g}) "
            f"= {pot_odds * 100:.1f}%. Equity ước tính ~{equity * 100:.1f}% {cmp_op} {pot_odds * 100:.1f}%.</step>\n"
            f"<step>4. Đọc vị: Quan sát lịch sử cược ở {obs.street}, range đối thủ cân bằng giữa value và bluff.</step>\n"
            f"<step>5. Quyết định: Dựa trên equity và vị trí {obs.position}, chọn hành động {action.value}.</step>\n"
            f"<action>{action.value}</action>"
        )
        parsed = parse_poker_cot(raw_text, legal_actions=obs.legal_actions)
        return AgentResult(
            action=parsed.resolved_action,
            raw_response=raw_text,
            prompt_text=prompt_text,
            parsed=parsed,
            metadata={"model": self.name},
        )
