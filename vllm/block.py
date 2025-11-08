"""Token blocks."""
from typing import TYPE_CHECKING, Iterator, List, Optional, Set

from vllm.utils import Device
from vllm.sequence import Sequence, SequenceGroup

import os
import math

_use_tanh = os.environ.get('USE_TANH', 'None') == '1'
_tanh_delta = os.environ.get('TANH_DELTA', 10)
_tanh_theta = os.environ.get('TANH_THETA', 1)
USE_TANH = False
if _use_tanh:
    try:
        TANH_DELTA = float(_tanh_delta)
    except:
        TANH_DELTA = 10
    try:
        TANH_THETA = float(_tanh_theta)
    except:
        TANH_THETA = 1
    USE_TANH = True

_half_life = os.environ.get('HALF_LIFE', 10)
try:
    HALF_LIFE = float(_half_life)
except ValueError:
    HALF_LIFE = 10

DEFAULT_LAST_ACCESSED_TIME: float = -1
BASE_FREQ: float = 0.5**(1/HALF_LIFE)


class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""

    def __init__(
        self,
        device: Device,
        block_number: int,
        block_size: int,
        block_hash: int,
        num_hashed_tokens: int,
    ) -> None:
        self.device = device
        self.block_number = block_number
        self.block_size = block_size
        self.block_hash = block_hash
        self.num_hashed_tokens = num_hashed_tokens

        self.ref_count = 0
        self.last_accessed = DEFAULT_LAST_ACCESSED_TIME
        # self.ref_seqs: Set[Sequence] = set()
        if USE_TANH:
            self.total_count = (0, 0)
        else:
            self.total_count = 0

        self.computed = False
        
        self.used_by_decode: bool = False # in prefill

        self.shared_with_prefill: bool = False

    def __repr__(self) -> str:
        return (f'PhysicalTokenBlock(device={self.device}, '
                f'block_number={self.block_number}, '
                f'num_hashed_tokens={self.num_hashed_tokens}, '
                f'ref_count={self.ref_count}, '
                f'last_accessed={self.last_accessed}, '
                f'computed={self.computed}), '
                f'total_count={self.total_count}')
    
    def add_total_count(self, count: int, now: float):
        if self.last_accessed != DEFAULT_LAST_ACCESSED_TIME:
            if self.USE_TANH:
                H, T = self.total_count
                last = self.last_accessed
                H *= math.exp(-(now - last) / TANH_THETA)
                T = 1 - 2 / ((2/(1-T)-1) * math.exp(-2*(now - last)/TANH_THETA) + 1)
                self.total_count = (H + T, math.tanh(TANH_DELTA / TANH_THETA) * count)
            else:
                self.total_count *= BASE_FREQ ** (now - self.last_accessed)
                self.total_count += count
    
    def get_total_count(self, now: Optional[float] = None) -> float:
        if self.USE_TANH:
            H, T = self.total_count
            last = self.last_accessed
            if now is None:
                return H + T
            return H * math.exp(-(now - last) / TANH_THETA) + \
                    1 - 2 / ((2/(1-T)-1) * math.exp(-2*(now - last)/TANH_THETA) + 1)
        else:
            return self.total_count * \
                (1 if now is None else BASE_FREQ ** (now - self.last_accessed))


class BlockTable:
    """Holds a list of blocks with caching of their associated block_ids 
    """

    def __init__(self, blocks: Optional[List[PhysicalTokenBlock]] = None):
        self._blocks: List[PhysicalTokenBlock] = []
        self._block_ids: List[int] = []

        if blocks is not None:
            for block in blocks:
                self.append(block)

    def get_blockids_from_hash(self, block_hash: int) -> List[int]:
        # find block
        block_ids = None
        for block in self._blocks:
            # print(f"block.block_hash: {block.block_hash}, block_hash: {block_hash}")
            if block.block_hash == block_hash:
                block_ids = block.block_number
                break
        return block_ids
    
    def append(self, block: PhysicalTokenBlock):
        self._blocks.append(block)
        self._block_ids.append(block.block_number)

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, key):
        return self._blocks[key]

    if TYPE_CHECKING:

        def __iter__(self) -> Iterator[PhysicalTokenBlock]:
            raise RuntimeError("Method should be automatically generated")

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            blocks = value
            self._blocks[key] = blocks
            self._block_ids[key] = [b.block_number for b in blocks]
        else:
            block = value
            self._blocks[key] = block
            self._block_ids[key] = block.block_number

    def reset(self):
        self._blocks = []
        self._block_ids = []

    def copy(self) -> "BlockTable":
        return BlockTable(self._blocks)

    def list(self) -> List[PhysicalTokenBlock]:
        return self._blocks

    def ids(self) -> List[int]:
        return self._block_ids
