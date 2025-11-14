"""Token blocks."""
from typing import TYPE_CHECKING, Iterator, List, Optional, Set

from vllm.utils import Device
from vllm.sequence import Sequence, SequenceGroup

import os
import math

_use_minexp = os.environ.get('USE_MINEXP', 'None') == '1'
_minexp_A = os.environ.get('MINEXP_A', 10)
_minexp_alpha = os.environ.get('MINEXP_ALPHA', 1)
_minexp_B = os.environ.get('MINEXP_B', 10)
_minexp_beta = os.environ.get('MINEXP_BETA', 1)

USE_MINEXP = False
if _use_minexp:
    USE_MINEXP = True
    try:
        MINEXP_ALPHA = float(_minexp_alpha)
    except:
        MINEXP_ALPHA = 1
    try:
        MINEXP_BETA = float(_minexp_beta)
    except:
        MINEXP_BETA = 1
    try:
        MINEXP_A = float(_minexp_A)
    except:
        MINEXP_A = 10
    try:
        MINEXP_B = float(_minexp_B)
    except:
        MINEXP_B = 10
    MINEXP_ALPHA = math.exp(-1 / MINEXP_ALPHA)
    MINEXP_A = MINEXP_ALPHA ** MINEXP_A
    MINEXP_BETA = math.exp(-1 / MINEXP_BETA)
    MINEXP_B = MINEXP_BETA ** MINEXP_B

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
        if USE_MINEXP:
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
            if self.USE_MINEXP:
                exp1, exp2 = self.total_count
                exp1 *= MINEXP_ALPHA ** (now - self.last_accessed)
                exp1 += MINEXP_A
                exp2 *= MINEXP_BETA ** (now - self.last_accessed)
                exp2 += MINEXP_B
                self.total_count = (exp1, exp2)
            else:
                self.total_count *= BASE_FREQ ** (now - self.last_accessed)
                self.total_count += count
    
    def get_total_count(self, now: Optional[float] = None) -> float:
        if self.USE_MINEXP:
            exp1, exp2 = self.total_count
            if now is None:
                return min(self.total_count)
            d = now - self.last_accessed
            return min(exp1 * (MINEXP_ALPHA ** d), exp2 * (MINEXP_BETA ** d))
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
