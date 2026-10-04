#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
from dataclasses import dataclass
from typing import Dict, List


DEFAULT_LOCAL_MODEL = "llm-jp/llm-jp-3-980m-instruct3"
DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"

SYSTEM_PROMPT = """あなたは日本の建築規制を扱うRAGアシスタントです。
次の規則を必ず守ってください。
1. 回答は、提示された参照資料だけを根拠にしてください。
2. 資料にない内容を推測したり、外部知識で補ったりしないでください。
3. 根拠が不足している場合は「提供された資料からは確認できません。」と回答してください。
4. 数値、地区名、資料名、例外条件は参照資料の表記を優先してください。
5. 回答は日本語で簡潔に書き、根拠となる文の末尾に [1] のような資料番号を付けてください。
6. 参照資料と矛盾する説明や、根拠のない資料番号は出力しないでください。"""


@dataclass(frozen=True)
class Citation:
    source_id: int
    document: str
    plan_number: str
    district: str
    item: str

    def format(self) -> str:
        fields = [self.document]
        if self.plan_number:
            fields.append(self.plan_number)
        if self.district:
            fields.append(self.district)
        if self.item:
            fields.append(self.item)
        return f"[{self.source_id}] " + " / ".join(fields)


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    citations: List[Citation]


def _metadata_value(chunk: str, field: str) -> str:
    match = re.search(rf"^{re.escape(field)}：(.+)$", chunk, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def build_citations(chunks: List[str]) -> List[Citation]:
    return [
        Citation(
            source_id=index,
            document=_metadata_value(chunk, "資料名"),
            plan_number=_metadata_value(chunk, "地区計画番号"),
            district=_metadata_value(chunk, "地区"),
            item=_metadata_value(chunk, "項目"),
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def build_numbered_context(chunks: List[str]) -> str:
    return "\n\n".join(
        f"【参照資料 {index}】\n{chunk}"
        for index, chunk in enumerate(chunks, start=1)
    )


class OpenAIChat:
    """OpenAI Responses API client for grounded Japanese RAG answers."""

    def __init__(self, model: str = DEFAULT_OPENAI_MODEL) -> None:
        from dotenv import find_dotenv, load_dotenv
        from openai import OpenAI

        load_dotenv(find_dotenv())
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to the project .env file "
                "before using --generator openai."
            )

        self.model = model
        self.client = OpenAI(api_key=api_key)

    def answer(self, question: str, chunks: List[str]) -> RAGAnswer:
        if not chunks:
            return RAGAnswer(
                answer="提供された資料からは確認できません。",
                citations=[],
            )

        user_prompt = (
            f"質問:\n{question}\n\n"
            f"参照資料:\n{build_numbered_context(chunks)}\n\n"
            "参照資料だけに基づいて回答してください。"
        )
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
        )
        answer = (response.output_text or "").strip()
        if not answer:
            answer = "提供された資料からは確認できません。"

        return RAGAnswer(answer=answer, citations=build_citations(chunks))


class LocalHFChat:
    """Local Hugging Face causal language model for grounded Japanese RAG answers."""

    def __init__(
        self,
        path: str = DEFAULT_LOCAL_MODEL,
        max_input_tokens: int = 3200,
        max_new_tokens: int = 256,
    ) -> None:
        self.path = path
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.load_model()

    def load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            torch_dtype=model_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _build_inputs(self, question: str, chunks: List[str]) -> Dict[str, object]:
        user_prompt = (
            f"質問:\n{question}\n\n"
            f"参照資料:\n{build_numbered_context(chunks)}\n\n"
            "参照資料だけに基づいて回答してください。"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        if self.tokenizer.chat_template:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
            )
        else:
            rendered_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\n回答:"
            inputs = self.tokenizer(
                rendered_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
            )

        return {
            name: inputs[name].to(self.device)
            for name in ("input_ids", "attention_mask")
            if name in inputs
        }

    def answer(self, question: str, chunks: List[str]) -> RAGAnswer:
        if not chunks:
            return RAGAnswer(
                answer="提供された資料からは確認できません。",
                citations=[],
            )

        inputs = self._build_inputs(question, chunks)
        input_length = inputs["input_ids"].shape[-1]

        with self._torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        answer = self.tokenizer.decode(
            output_ids[0][input_length:],
            skip_special_tokens=True,
        ).strip()
        if not answer:
            answer = "提供された資料からは確認できません。"

        return RAGAnswer(answer=answer, citations=build_citations(chunks))
