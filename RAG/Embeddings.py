#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List

import numpy as np


class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError

    def get_document_embedding(self, text: str) -> List[float]:
        return self.get_embedding(text)

    def get_query_embedding(self, text: str) -> List[float]:
        return self.get_embedding(text)
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
    

class RuriV3Embedding(BaseEmbeddings):
    """
    Local Japanese Ruri v3 embeddings with retrieval-specific prefixes.
    """
    def __init__(
        self,
        path: str = 'cl-nagoya/ruri-v3-130m',
        is_api: bool = False,
    ) -> None:
        super().__init__(path, is_api)

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.document_prefix = "検索文書: "
        self.query_prefix = "検索クエリ: "
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.path)
        self._model = AutoModel.from_pretrained(self.path).to(self._device)
        self._model.eval()

    def _average_pool(self, last_hidden_state, attention_mask):
        masked_hidden_state = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(),
            0.0,
        )
        token_count = attention_mask.sum(dim=1)[..., None].clamp(min=1)
        return masked_hidden_state.sum(dim=1) / token_count

    def _encode(self, text: str) -> List[float]:
        encoded_input = self._tokenizer(
            [text],
            max_length=8192,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded_input = {
            key: value.to(self._device)
            for key, value in encoded_input.items()
        }

        with self._torch.no_grad():
            model_output = self._model(**encoded_input)
            embedding = self._average_pool(
                model_output.last_hidden_state,
                encoded_input["attention_mask"],
            )
            embedding = self._torch.nn.functional.normalize(
                embedding,
                p=2,
                dim=1,
            )

        return embedding[0].cpu().tolist()

    def get_embedding(self, text: str) -> List[float]:
        return self._encode(text)

    def get_document_embedding(self, text: str) -> List[float]:
        return self._encode(self.document_prefix + text)

    def get_query_embedding(self, text: str) -> List[float]:
        return self._encode(self.query_prefix + text)
