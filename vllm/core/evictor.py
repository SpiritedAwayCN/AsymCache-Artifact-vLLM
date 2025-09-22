# SPDX-License-Identifier: Apache-2.0

import enum
import heapq
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

from vllm.logger import init_logger
logger = init_logger(__name__)


class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()
    MWF = enum.auto()


class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed Blocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        pass

    @abstractmethod
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, total_count: Optional[float] = None):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        pass

    @abstractmethod
    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass


class BlockMetaData:
    """Data structure for storing key data describe cached block, so that
    evitor could use to make its decision which one to choose for eviction

    Here we use physical block id as the dict key, as there maybe several
    blocks with the same content hash, but their physical id is unique.
    """

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float, total_count: Optional[float] = None):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed
        self.total_count = total_count if total_count is not None else 0.0


class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the Block. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    # CLEANUP_THRESHOLD determines the maximum allowable size of the priority
    # queue relative to the free table size. When this threshold is exceeded,
    # a cleanup operation is triggered to reduce memory usage.
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []
        self.protect_blocks = set()

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        tmp = []
        while self.priority_queue:
            # We do not remove outdated entries from the priority queue at the
            # time of updating the last_accessed timestamp. Instead, outdated
            # entries are filtered out here during eviction. Outdated entries
            # would either not in the free table, or have older last accessed
            # time.
            last_accessed, v, block_id, content_hash = heapq.heappop(
                self.priority_queue)
            if block_id in self.protect_blocks:
                tmp.append((last_accessed, v, block_id, content_hash))
                continue
            if (block_id in self.free_table and
                    self.free_table[block_id].last_accessed == last_accessed):
                self.free_table.pop(block_id)
                import time
                now = time.time()
                for _, v, b, c in tmp:
                    heapq.heappush(self.priority_queue, (now, v, b, c))
                    if b in self.free_table:
                        self.free_table[b].last_accessed = now
                # print(f"protect={self.protect_blocks}")
                logger.info(f"Evicting block_id {block_id} with content_hash {content_hash}, last_accessed {last_accessed}, num_hashed_tokens {-v}")
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, total_count: Optional[float] = None):
        import os
        if os.getenv('EVCITOR_BLOCK_PROTECT_FLAG') == '1':
            self.protect_blocks.add(block_id)
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed)
        heapq.heappush(
            self.priority_queue,
            (last_accessed, -num_hashed_tokens, block_id, content_hash))
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > LRUEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[float, int, int, int]] = []

        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.last_accessed, -block.num_hashed_tokens, block_id,
                 block.content_hash))
        heapq.heapify(new_priority_queue)

        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        return self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

class WeightEvictor(Evictor):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed Blocks.
    """
    def __init__(self, base, delta_func):
        # self.delta(len) ---> delta_T
        self.base = base
        self.delta = delta_func
        # block_hash ---> (block, weight)
        self.free_table: Dict[int, BlockMetaData] = dict()

        import sys
        import os
        module_directory = os.environ.get("CSET_DIR", None)
        if module_directory is None:
            logger.error("CSET_DIR environment variable not set")
        else:
            logger.info(f"Using CSET_DIR: {module_directory}")
        sys.path.append(module_directory)
        from c_set import c_set
        self.heap = c_set(base)
        self._has_evicted = False

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        evicted_id = self.heap.pop_first()
        evicted_block = self.free_table.pop(evicted_id)
        if not self._has_evicted:
            self._has_evicted = True
            logger.info(f"First eviction, heap size = {len(self.free_table)}")
        return evicted_id, evicted_block.content_hash

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, total_count: Optional[float] = None):
        """Adds block to the evictor, making it a candidate for eviction"""
        self.free_table[block_id] = BlockMetaData(
            content_hash, num_hashed_tokens, last_accessed, total_count)
        self.heap.insert(total_count * self.delta(num_hashed_tokens), last_accessed, block_id)

    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        logger.warning(f"WeightEvictor does not support update operation, got block_id={block_id}, last_accessed={last_accessed}")
        block = self.free_table.get(block_id)
        if block is None:
            raise ValueError(f"Block {block_id} not found in free_table")
        self.heap.erase(block.total_count * self.delta(block.num_hashed_tokens), block.last_accessed, block_id)
        
        block.total_count *= self.base ** (last_accessed - block.last_accessed)
        block.total_count += 1
        block.last_accessed = last_accessed
        # Reinsert the block with the updated last_accessed time
        self.heap.insert(block.total_count * self.delta(block.num_hashed_tokens), last_accessed, block_id)  

    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block = self.free_table.pop(block_id)
        self.heap.erase(block.total_count * self.delta(block.num_hashed_tokens), block.last_accessed, block_id)
        return block

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        print("evictor:LRU")
        return LRUEvictor()
    elif eviction_policy == EvictionPolicy.MWF:
        from vllm.core.block.common import BASE_FREQ
        c = 2.91200787e-01 + 1.04221165e-04
        k = 2 * 1.04221165e-04 / 16
        return WeightEvictor(BASE_FREQ, lambda x: k * x + c)
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
