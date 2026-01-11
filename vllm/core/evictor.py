# SPDX-License-Identifier: Apache-2.0

import enum
import heapq
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
from vllm.logger import init_logger
logger = init_logger(__name__)
# from vllm.utils import func_timer

class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()
    MWF = enum.auto()
    LFU = enum.auto()
    MWF_BS = enum.auto()
    BS = enum.auto()
    PENSIEVE = enum.auto()


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
            last_accessed: float, total_count: Optional[Union[float, Tuple[float, float]]] = None):
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
                 last_accessed: float, total_count: Optional[Union[float, Tuple[float, float]]] = None,
                 func: Optional[float] = None):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed
        if total_count is None:
            from vllm.core.block.common import USE_MINEXP
            self.total_count = (0, 0) if USE_MINEXP else 0
        else:
            self.total_count = total_count
        self.func = func if func is not None else 0
    def __repr__(self) -> str:
        return (f'BlockMetaData,'
                f'num_hashed_tokens={self.num_hashed_tokens}, '
                f'last_accessed={self.last_accessed}, '
                f'total_count={self.total_count}, '
                f'func={self.func}')

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
                # logger.info(f"Evicting block_id {block_id} with content_hash {content_hash}, last_accessed {last_accessed}, num_hashed_tokens {-v}")
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, total_count: Optional[Union[float, Tuple[float, float]]] = None):
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

class LFUEvictor(Evictor):
    
    def __init__(self, param1, param2 = None, param3 = None, param4 = None):
        if param2 is None:
            logger.info(f"{self.__class__.__name__} is used, base is {param1}.")
            # block_hash ---> (block, weight)
            self.free_table: Dict[int, BlockMetaData] = dict()
            self.USE_MINEXP = False
            self.base = param1
            import sys
            import os
            module_directory = os.environ.get("CSET_DIR", None)
            if not os.path.exists(module_directory):
                raise FileNotFoundError(f"Module directory {module_directory} not found!")
            sys.path.append(module_directory)
            try:
                from c_set import c_set_lfu  # 导入 c_set_lfu 模块
            except ImportError as e:
                raise ImportError(f"Failed to import 'c_set_lfu' from {module_directory}: {e}")
            self.heap = c_set_lfu(self.base)
            self._has_evicted = False
        else:
            logger.info(f"{self.__class__.__name__}_minexp<{param1}, {param2}, {param3}, {param4}> is used")
            # block_hash ---> (block, weight)
            self.free_table: Dict[int, BlockMetaData] = dict()
            self.USE_MINEXP = True
            self.MINEXP_A = param1
            self.MINEXP_B = param2
            self.MINEXP_ALPHA = param3
            self.MINEXP_BETA = param4
            import sys
            import os
            module_directory = os.environ.get("CSET_DIR", None)
            if not os.path.exists(module_directory):
                raise FileNotFoundError(f"Module directory {module_directory} not found!")
            sys.path.append(module_directory)
            try:
                from c_set import c_set_lfu  # 导入 c_set_lfu 模块
            except ImportError as e:
                raise ImportError(f"Failed to import 'c_set_lfu' from {module_directory}: {e}")
            self.heap1 = c_set_lfu(self.MINEXP_ALPHA)
            self.heap2 = c_set_lfu(self.MINEXP_BETA)
            self._has_evicted = False

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")
        if not self.USE_MINEXP:
            # logger.info(f"evicting, heap size = {self.heap.size()}")
            evicted_id = self.heap.pop_first()
            # block = self.free_table[evicted_id]
            # print(f"evict:{block.total_count}")
            evicted_block = self.free_table.pop(evicted_id)
        else:
            weight1, last_accessed1 = self.heap1.first()
            weight2, last_accessed2 = self.heap2.first()
            if last_accessed1 < last_accessed2:
                weight1 *= self.MINEXP_ALPHA ** (last_accessed2 - last_accessed1)
            else:
                weight2 *= self.MINEXP_BETA ** (last_accessed1 - last_accessed2)
            if weight1 < weight2:
                evicted_id = self.heap1.pop_first()
                evicted_block = self.free_table.pop(evicted_id)
                self.heap2.erase(evicted_block.total_count[1], evicted_block.last_accessed, evicted_id, evicted_block.num_hashed_tokens)
            else:
                evicted_id = self.heap2.pop_first()
                evicted_block = self.free_table.pop(evicted_id)
                self.heap1.erase(evicted_block.total_count[0], evicted_block.last_accessed, evicted_id, evicted_block.num_hashed_tokens)
        if not self._has_evicted:
            self._has_evicted = True
            logger.info(f"First eviction, heap size = {len(self.free_table)}")
        return evicted_id, evicted_block.content_hash

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, total_count: Optional[Union[float, Tuple[float, float]]] = None):
        self.free_table[block_id] = BlockMetaData(
            content_hash, num_hashed_tokens, last_accessed, total_count)
        if self.USE_MINEXP:
            self.heap1.insert(total_count[0], last_accessed, block_id, num_hashed_tokens)
            self.heap2.insert(total_count[1], last_accessed, block_id, num_hashed_tokens)
        else:
            self.heap.insert(total_count, last_accessed, block_id, num_hashed_tokens)

    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        block = self.free_table.get(block_id)
        if block is None:
            raise ValueError(f"Block {block_id} not found in free_table")
        if self.USE_MINEXP:
            self.heap1.erase(block.total_count[0], block.last_accessed, block_id, block.num_hashed_tokens)
            self.heap2.erase(block.total_count[1], block.last_accessed, block_id, block.num_hashed_tokens)
            exp1, exp2 = block.total_count
            exp1 *= self.MINEXP_ALPHA ** (now - self.last_accessed)
            exp1 += self.MINEXP_A
            exp2 *= self.MINEXP_BETA ** (now - self.last_accessed)
            exp2 += self.MINEXP_B
            block.total_count = (exp1, exp2)
            block.last_accessed = last_accessed
            self.heap1.insert(block.total_count[0], block.last_accessed, block_id, block.num_hashed_tokens)
            self.heap2.insert(block.total_count[1], block.last_accessed, block_id, block.num_hashed_tokens)
        else:
            self.heap.erase(block.total_count, block.last_accessed, block_id, block.num_hashed_tokens)
            block.total_count *= self.base ** (last_accessed - block.last_accessed)
            block.total_count += 1
            block.last_accessed = last_accessed
            # Reinsert the block with the updated last_accessed time
            self.heap.insert(block.total_count, block.last_accessed, block_id, block.num_hashed_tokens)

    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block = self.free_table.pop(block_id)
        if self.USE_MINEXP:
            self.heap1.erase(block.total_count[0], block.last_accessed, block_id, block.num_hashed_tokens)
            self.heap2.erase(block.total_count[1], block.last_accessed, block_id, block.num_hashed_tokens)
        else:
            self.heap.erase(block.total_count, block.last_accessed, block_id, block.num_hashed_tokens)

        return block

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)
class MWF_BSEvictor(Evictor):
    def __init__(self, param1, param2, param3, param4, param5):
        self.free_table: Dict[int, BlockMetaData] = dict()
        self.MINEXP_A = param1
        self.MINEXP_B = param2
        self.MINEXP_ALPHA = param3
        self.MINEXP_BETA = param4
        self.delta = param5

        import sys
        import os
        module_directory = os.environ.get("CSET_DIR", None)
        if module_directory is None:
            looger.error("CSET_DIR environment variable not set")
        else:
            logger.info(f"Using c_list in CSET_DIR: {module_directory}")
        sys.path.append(module_directory)
        from c_set import c_list
        self.list1 = c_list(self.MINEXP_ALPHA)
        self.list2 = c_list(self.MINEXP_BETA)
        self._has_evicted = False

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        weight1, last_accessed1 = self.list1.min()
        weight2, last_accessed2 = self.list2.min()
        if last_accessed1 < last_accessed2:
            weight1 *= self.MINEXP_ALPHA ** (last_accessed2 - last_accessed1)
        else:
            weight2 *= self.MINEXP_BETA ** (last_accessed1 - last_accessed2)
        if weight1 < weight2:
            evicted_id = self.list1.pop_min()
            evicted_block = self.free_table.pop(evicted_id)
            self.list2.erase(evicted_block.total_count[1] * self.delta(evicted_block.num_hashed_tokens), 
                                evicted_block.last_accessed, evicted_id)
        else:
            evicted_id = self.list2.pop_min()
            evicted_block = self.free_table.pop(evicted_id)
            self.list1.erase(evicted_block.total_count[0] * self.delta(evicted_block.num_hashed_tokens), 
                                evicted_block.last_accessed, evicted_id)
        
        if not self._has_evicted:
            self._has_evicted = True
            logger.info(f"First eviction, list size = {len(self.free_table)}")
        return evicted_id, evicted_block.content_hash

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, total_count: Optional[Union[float, Tuple[float, float]]] = None):
        """Adds block to the evictor, making it a candidate for eviction"""
        # assert block_id not in self.free_table, f"Block {block_id} already in evictor"

        val = self.delta(num_hashed_tokens)
        self.list1.insert(total_count[0] * val, last_accessed, block_id)
        self.list2.insert(total_count[1] * val, last_accessed, block_id)
    
        self.free_table[block_id] = BlockMetaData(
            content_hash, num_hashed_tokens, last_accessed, total_count, val)

    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        logger.warning(f"MWF_BSEvictor does not support update operation, got block_id={block_id}, last_accessed={last_accessed}")
        block = self.free_table.get(block_id)
        if block is None:
            raise ValueError(f"Block {block_id} not found in free_table")
        val = self.delta(block.num_hashed_tokens)
        self.list1.erase(block.total_count[0] * val, block.last_accessed, block_id)
        self.list2.erase(block.total_count[1] * val, block.last_accessed, block_id)
        exp1, exp2 = block.total_count
        exp1 *= self.MINEXP_ALPHA ** (now - self.last_accessed)
        exp1 += self.MINEXP_A
        exp2 *= self.MINEXP_BETA ** (now - self.last_accessed)
        exp2 += self.MINEXP_B
        block.total_count = (exp1, exp2)
        block.last_accessed = last_accessed
        self.list1.insert(block.total_count[0] * val, block.last_accessed, block_id)
        self.list2.insert(block.total_count[1] * val, block.last_accessed, block_id)

    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block = self.free_table.pop(block_id)
        val = self.delta(block.num_hashed_tokens)
        self.list1.erase(block.total_count[0] * val, block.last_accessed, block_id)
        self.list2.erase(block.total_count[1] * val, block.last_accessed, block_id)
        return block

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

class BSEvictor(Evictor):
    def __init__(self, param1, param2, param3, param4):
        self.free_table: Dict[int, BlockMetaData] = dict()
        self.MINEXP_A = param1
        self.MINEXP_B = param2
        self.MINEXP_ALPHA = param3
        self.MINEXP_BETA = param4

        import sys
        import os
        module_directory = os.environ.get("CSET_DIR", None)
        if module_directory is None:
            looger.error("CSET_DIR environment variable not set")
        else:
            logger.info(f"Using c_list in CSET_DIR: {module_directory}")
        sys.path.append(module_directory)
        from c_set import c_list_with_num
        self.list1 = c_list_with_num(self.MINEXP_ALPHA)
        self.list2 = c_list_with_num(self.MINEXP_BETA)
        self._has_evicted = False

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        weight1, last_accessed1 = self.list1.min()
        weight2, last_accessed2 = self.list2.min()
        if last_accessed1 < last_accessed2:
            weight1 *= self.MINEXP_ALPHA ** (last_accessed2 - last_accessed1)
        else:
            weight2 *= self.MINEXP_BETA ** (last_accessed1 - last_accessed2)
        if weight1 < weight2:
            evicted_id = self.list1.pop_min()
            evicted_block = self.free_table.pop(evicted_id)
            self.list2.erase(evicted_block.total_count[1], 
                                evicted_block.last_accessed, evicted_id, evicted_block.num_hashed_tokens)
        else:
            evicted_id = self.list2.pop_min()
            evicted_block = self.free_table.pop(evicted_id)
            self.list1.erase(evicted_block.total_count[0], 
                                evicted_block.last_accessed, evicted_id, evicted_block.num_hashed_tokens)
        
        if not self._has_evicted:
            self._has_evicted = True
            logger.info(f"First eviction, list size = {len(self.free_table)}")
        return evicted_id, evicted_block.content_hash

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, total_count: Optional[Union[float, Tuple[float, float]]] = None):
        """Adds block to the evictor, making it a candidate for eviction"""
        # assert block_id not in self.free_table, f"Block {block_id} already in evictor"

        self.list1.insert(total_count[0], last_accessed, block_id, num_hashed_tokens)
        self.list2.insert(total_count[1], last_accessed, block_id, num_hashed_tokens)
    
        self.free_table[block_id] = BlockMetaData(
            content_hash, num_hashed_tokens, last_accessed, total_count)

    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        logger.warning(f"BSEvictor does not support update operation, got block_id={block_id}, last_accessed={last_accessed}")
        block = self.free_table.get(block_id)
        if block is None:
            raise ValueError(f"Block {block_id} not found in free_table")
        self.list1.erase(block.total_count[0], block.last_accessed, block_id, block.num_hashed_tokens)
        self.list2.erase(block.total_count[1], block.last_accessed, block_id, block.num_hashed_tokens)
        exp1, exp2 = block.total_count
        exp1 *= self.MINEXP_ALPHA ** (now - self.last_accessed)
        exp1 += self.MINEXP_A
        exp2 *= self.MINEXP_BETA ** (now - self.last_accessed)
        exp2 += self.MINEXP_B
        block.total_count = (exp1, exp2)
        block.last_accessed = last_accessed
        self.list1.insert(block.total_count[0], block.last_accessed, block_id, block.num_hashed_tokens)
        self.list2.insert(block.total_count[1], block.last_accessed, block_id, block.num_hashed_tokens)

    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block = self.free_table.pop(block_id)
        self.list1.erase(block.total_count[0], block.last_accessed, block_id, block.num_hashed_tokens)
        self.list2.erase(block.total_count[1], block.last_accessed, block_id, block.num_hashed_tokens)
        return block

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)
class PensieveEvictor(Evictor):
    def __init__(self, param1):
        self.free_table: Dict[int, BlockMetaData] = dict()
        self.block_list: List[int] = []
        self._has_evicted = False
        self.delta = param1

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def _test_display(self):
        temp = sorted([self.free_table[blk].total_count for blk in self.block_list])
        print(f"[PensieveEvictor total_access]: {temp[:10]+temp[-10:]}")
        temp = sorted([self.free_table[blk].func/self.free_table[blk].total_count for blk in self.block_list])
        print(f"[PensieveEvictor deltaT/total_access]: {temp[:10]+temp[-10:]}")

    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")
        chosen = -1
        chosen_val = 0
        for blk_id in self.block_list:
            blk_val = self.free_table[blk_id].func / self.free_table[blk_id].total_count
            if chosen == -1 or blk_val < chosen_val:
                chosen = blk_id
                chosen_val = blk_val
        
        # self._test_display()
        # print(f"choose val={chosen_val}")
        evicted_id = chosen
        evicted_block = self.free_table.pop(evicted_id)
        self.block_list.remove(evicted_id)
        if not self._has_evicted:
            self._has_evicted = True
            logger.info(f"First eviction, heap size = {len(self.free_table)}")
        return evicted_id, evicted_block.content_hash

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, total_count: int):
        """Adds block to the evictor, making it a candidate for eviction"""
        # assert block_id not in self.free_table, f"Block {block_id} already in evictor"

        val = self.delta(num_hashed_tokens)
        self.block_list.append(block_id)
    
        self.free_table[block_id] = BlockMetaData(
            content_hash, num_hashed_tokens, last_accessed, total_count, val)

    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        logger.warning(f"PensieveEvictor does not support update operation, got block_id={block_id}, last_accessed={last_accessed}")
        raise NotImplementedError

    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block = self.free_table.pop(block_id)
        self.block_list.remove(block_id)
        return block

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class WeightEvictor(Evictor):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed Blocks.
    """
    def __init__(self, param1, param2, param3 = None, param4 = None, param5 = None):
        # self.delta(len) ---> delta_T
        if param3 is None:
            self.base = param1
            self.delta = param2
            self.USE_MINEXP = False
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
            self.heap = c_set(self.base)
            self._has_evicted = False
        else:
            self.MINEXP_A = param1
            self.MINEXP_B = param2
            self.MINEXP_ALPHA = param3
            self.MINEXP_BETA = param4
            self.delta = param5
            self.USE_MINEXP = True
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
            self.heap1 = c_set(self.MINEXP_ALPHA)
            self.heap2 = c_set(self.MINEXP_BETA)
            self._has_evicted = False

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        if not self.USE_MINEXP:
            evicted_id = self.heap.pop_first()
            evicted_block = self.free_table.pop(evicted_id)
        else:
            weight1, last_accessed1 = self.heap1.first()
            weight2, last_accessed2 = self.heap2.first()
            if last_accessed1 < last_accessed2:
                weight1 *= self.MINEXP_ALPHA ** (last_accessed2 - last_accessed1)
            else:
                weight2 *= self.MINEXP_BETA ** (last_accessed1 - last_accessed2)
            if weight1 < weight2:
                evicted_id = self.heap1.pop_first()
                evicted_block = self.free_table.pop(evicted_id)
                self.heap2.erase(evicted_block.total_count[1] * self.delta(evicted_block.num_hashed_tokens), 
                                 evicted_block.last_accessed, evicted_id)
            else:
                evicted_id = self.heap2.pop_first()
                evicted_block = self.free_table.pop(evicted_id)
                self.heap1.erase(evicted_block.total_count[0] * self.delta(evicted_block.num_hashed_tokens), 
                                 evicted_block.last_accessed, evicted_id)
            
        if not self._has_evicted:
            self._has_evicted = True
            logger.info(f"First eviction, heap size = {len(self.free_table)}")
        return evicted_id, evicted_block.content_hash

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, total_count: Optional[Union[float, Tuple[float, float]]] = None):
        """Adds block to the evictor, making it a candidate for eviction"""
        # assert block_id not in self.free_table, f"Block {block_id} already in evictor"
        if not self.USE_MINEXP:
            val = total_count * self.delta(num_hashed_tokens)
            self.heap.insert(val, last_accessed, block_id)
        else:
            val = self.delta(num_hashed_tokens)
            self.heap1.insert(total_count[0] * val, last_accessed, block_id)
            self.heap2.insert(total_count[1] * val, last_accessed, block_id)
        
        self.free_table[block_id] = BlockMetaData(
            content_hash, num_hashed_tokens, last_accessed, total_count, val)

    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        logger.warning(f"WeightEvictor does not support update operation, got block_id={block_id}, last_accessed={last_accessed}")
        block = self.free_table.get(block_id)
        if block is None:
            raise ValueError(f"Block {block_id} not found in free_table")
        val = self.delta(block.num_hashed_tokens)
        if self.USE_MINEXP:
            self.heap1.erase(block.total_count[0] * val, block.last_accessed, block_id)
            self.heap2.erase(block.total_count[1] * val, block.last_accessed, block_id)
            exp1, exp2 = block.total_count
            exp1 *= self.MINEXP_ALPHA ** (now - self.last_accessed)
            exp1 += self.MINEXP_A
            exp2 *= self.MINEXP_BETA ** (now - self.last_accessed)
            exp2 += self.MINEXP_B
            block.total_count = (exp1, exp2)
            block.last_accessed = last_accessed
            self.heap1.insert(block.total_count[0] * val, block.last_accessed, block_id)
            self.heap2.insert(block.total_count[1] * val, block.last_accessed, block_id)
        else:
            self.heap.erase(block.total_count * val, block.last_accessed, block_id)
            block.total_count *= self.base ** (last_accessed - block.last_accessed)
            block.total_count += 1
            block.last_accessed = last_accessed
            # Reinsert the block with the updated last_accessed time
            self.heap.insert(block.total_count * val, last_accessed, block_id) 

    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block = self.free_table.pop(block_id)
        val = self.delta(block.num_hashed_tokens)
        if self.USE_MINEXP:
            self.heap1.erase(block.total_count[0] * val, block.last_accessed, block_id)
            self.heap2.erase(block.total_count[1] * val, block.last_accessed, block_id)
        else:
            self.heap.erase(block.total_count * val, block.last_accessed, block_id)
        return block

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        print("evictor:LRU")
        return LRUEvictor()
    elif eviction_policy == EvictionPolicy.MWF:
        from vllm.core.block.common import BASE_FREQ, USE_MINEXP
        c = 16.975203511045493
        k = 0.00043533405552524906
        if USE_MINEXP:
            print("evictor:MWF_MINEXP")
            from vllm.core.block.common import MINEXP_A, MINEXP_B, MINEXP_ALPHA, MINEXP_BETA
            return WeightEvictor(MINEXP_A, MINEXP_B, MINEXP_ALPHA, MINEXP_BETA, lambda x: k * x + c)
        else:
            return WeightEvictor(BASE_FREQ, lambda x: k * x + c)
    elif eviction_policy == EvictionPolicy.LFU:
        from vllm.core.block.common import BASE_FREQ, USE_MINEXP
        print("evictor:LFU")
        if USE_MINEXP:
            print("evictor:LFU_MINEXP")
            from vllm.core.block.common import MINEXP_A, MINEXP_B, MINEXP_ALPHA, MINEXP_BETA
            return LFUEvictor(MINEXP_A, MINEXP_B, MINEXP_ALPHA, MINEXP_BETA)
        else:
            return LFUEvictor(BASE_FREQ)
    elif eviction_policy == EvictionPolicy.MWF_BS:
        from vllm.core.block.common import BASE_FREQ, USE_MINEXP
        assert USE_MINEXP
        print("evictor:MWF_BS")
        c = 1.8544166376320397
        k = 7.411805757842883e-05
        from vllm.core.block.common import MINEXP_A, MINEXP_B, MINEXP_ALPHA, MINEXP_BETA
        return MWF_BSEvictor(MINEXP_A, MINEXP_B, MINEXP_ALPHA, MINEXP_BETA, lambda x: k * x + c)
    elif eviction_policy == EvictionPolicy.BS:
        from vllm.core.block.common import BASE_FREQ, USE_MINEXP
        assert USE_MINEXP
        print("evictor:BS")
        from vllm.core.block.common import MINEXP_A, MINEXP_B, MINEXP_ALPHA, MINEXP_BETA
        return BSEvictor(MINEXP_A, MINEXP_B, MINEXP_ALPHA, MINEXP_BETA)        
    elif eviction_policy == EvictionPolicy.PENSIEVE:
        print("evictor:PensieveEvictor")
        c = 1.8544166376320397
        k = 7.411805757842883e-05
        return PensieveEvictor(lambda x: k * x + c)
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
