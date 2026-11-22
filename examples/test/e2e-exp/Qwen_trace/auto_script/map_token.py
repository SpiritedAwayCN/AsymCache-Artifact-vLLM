#!/usr/bin/env python3
"""Deterministic mapping from Trace A hash blocks to tokenizer token IDs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MASK64 = (1 << 64) - 1


def splitmix64(value: int) -> int:
    """A bijective 64-bit mixer used before base-vocabulary encoding."""
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def expand_ranges(ranges: list[list[int]]) -> list[int]:
    values: list[int] = []
    for start, end in ranges:
        if start > end:
            raise ValueError(f"Invalid token ID range: {start}..{end}")
        values.extend(range(int(start), int(end) + 1))
    return values


class TraceTokenMapper:
    def __init__(self, config: dict[str, Any]) -> None:
        if config.get("algorithm") != "splitmix64_base_vocab_v1":
            raise ValueError(f"Unsupported mapping algorithm: {config.get('algorithm')}")
        self.seed = int(config["mapping_seed"]) & MASK64
        self.trace_block_size = int(config["trace_hash_block_size"])
        self.token_ids = expand_ranges(config["safe_token_id_ranges"])
        self.base = len(self.token_ids)
        if self.base != int(config["safe_token_count"]):
            raise ValueError("safe_token_count does not match safe_token_id_ranges")

        capacity = 1
        self.unique_positions = 0
        while capacity <= MASK64:
            capacity *= self.base
            self.unique_positions += 1
        if self.unique_positions > self.trace_block_size:
            raise ValueError("Vocabulary is too small for an injective block mapping")

    @classmethod
    def from_path(cls, path: Path) -> "TraceTokenMapper":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def tokens_for_hash(self, hash_id: int) -> list[int]:
        if not 0 <= hash_id <= MASK64:
            raise ValueError(f"hash_id is outside uint64: {hash_id}")

        mixed = splitmix64(hash_id ^ self.seed)
        value = mixed
        result: list[int] = []
        for _ in range(self.unique_positions):
            result.append(self.token_ids[value % self.base])
            value //= self.base

        state = mixed
        for position in range(self.unique_positions, self.trace_block_size):
            state = splitmix64(
                state
                ^ hash_id
                ^ ((position + 1) * 0xD6E8FEB86659FD93 & MASK64)
            )
            result.append(self.token_ids[state % self.base])
        return result

    def prompt_token_ids(
        self, hash_ids: list[int], input_length: int
    ) -> list[int]:
        return self.prompt_slice(hash_ids, input_length, 0, input_length)

    def prompt_slice(
        self,
        hash_ids: list[int],
        input_length: int,
        start: int,
        end: int,
    ) -> list[int]:
        start = max(0, min(int(start), input_length))
        end = max(start, min(int(end), input_length))
        if start == end:
            return []

        first_block = start // self.trace_block_size
        last_block = (end - 1) // self.trace_block_size
        result: list[int] = []
        for block_index in range(first_block, last_block + 1):
            result.extend(self.tokens_for_hash(int(hash_ids[block_index])))
        left_trim = start - first_block * self.trace_block_size
        return result[left_trim : left_trim + end - start]

    def synthetic_output_ids(self, chat_id: int, length: int) -> list[int]:
        result: list[int] = []
        block_index = 0
        while len(result) < length:
            domain_value = (
                (int(chat_id) * 0x9E3779B97F4A7C15)
                ^ (block_index * 0xD6E8FEB86659FD93)
                ^ 0x4F5554505554
            ) & MASK64
            result.extend(self.tokens_for_hash(splitmix64(domain_value)))
            block_index += 1
        return result[:length]
