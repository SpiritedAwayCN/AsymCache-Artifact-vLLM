import enum
from abc import ABC, abstractmethod
from typing import OrderedDict

from vllm.block import PhysicalTokenBlock
from vllm.logger import init_logger

logger = init_logger(__name__)

timestamp_thres_nblock = 1000
prefix_thres_tokens = 96

class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()
    LRU2 = enum.auto()
    LRU3 = enum.auto()


class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed PhysicalTokenBlocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_hash: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> PhysicalTokenBlock:
        """Runs the eviction algorithm and returns the evicted block"""
        pass

    @abstractmethod
    def add(self, block: PhysicalTokenBlock):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        """Simply removes the block with the hash value block_hash from the
        evictor. Caller is responsible for making sure that block_hash is
        contained in the evictor before calling remove. Should be used to
        "bring back" blocks that have been freed but not evicted yet.
        """
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass


class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the PhysicalTokenBlock. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    def __init__(self):
        self.free_table: OrderedDict[int, PhysicalTokenBlock] = OrderedDict()

    def __contains__(self, block_hash: int) -> bool:
        return block_hash in self.free_table

    def evict(self) -> PhysicalTokenBlock:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        evicted_block = None
        # next(iter(self.free_table.values()))
        # The blocks with the lowest timestamps should be placed consecutively
        # at the start of OrderedDict. Loop through all these blocks to
        # find the one with maximum number of hashed tokens.
        
        # temp implementation
        # logger.info(f"num free table: {len(self.free_table)}")
        for _, block in self.free_table.items():
            # logger.info(f"Block {block.block_number} is being considered for eviction, ")
                        # f"ref_seqs: {block.ref_seqs}")
            for seq in block.ref_seqs:
                if seq.slo_urgent and seq.slo_urgent >= 1:
                    # logger.info(f"Block {block.block_number} is not evicted "
                    #             f"because of slo_urgent from seq {seq}")
                    break
            else:
                if evicted_block is None:
                    evicted_block = block
                    continue
                if evicted_block.last_accessed < block.last_accessed:
                    break
                if evicted_block.num_hashed_tokens < block.num_hashed_tokens:
                    evicted_block = block

        self.free_table.pop(evicted_block.block_hash)
        
        # logger.info(f'Block {evicted_block.block_number} is evicted') #, ref_seqs: {evicted_block.ref_seqs}')
        evicted_block.computed = False
        evicted_block.ref_seqs.clear()
        return evicted_block

    def add(self, block: PhysicalTokenBlock):
        self.free_table[block.block_hash] = block

    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        if block_hash not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block: PhysicalTokenBlock = self.free_table[block_hash]
        self.free_table.pop(block_hash)
        return block

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

class LRU3Evictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the PhysicalTokenBlock. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    def __init__(self):
        logger.info(f"{self.__class__.__name__} is used.")
        self.has_evicted = False
        self.free_table: OrderedDict[int, PhysicalTokenBlock] = OrderedDict()

    def __contains__(self, block_hash: int) -> bool:
        return block_hash in self.free_table

    def evict(self) -> PhysicalTokenBlock:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        evicted_block = next(iter(self.free_table.values()))
        # The blocks with the lowest timestamps should be placed consecutively
        # at the start of OrderedDict. Loop through all these blocks to
        # find the one with maximum number of hashed tokens.
        visited = 0
        for _, block in self.free_table.items():
            # if not self.has_evicted:
            #     logger.info(f"Block {block.block_number} (nht={block.num_hashed_tokens}) is being considered for eviction.")
            visited += 1
            if evicted_block.last_accessed < block.last_accessed and visited > timestamp_thres_nblock:
                break
            if evicted_block.num_hashed_tokens > prefix_thres_tokens:
                if prefix_thres_tokens < block.num_hashed_tokens < evicted_block.num_hashed_tokens:
                    evicted_block = block
            elif block.num_hashed_tokens > prefix_thres_tokens or \
                    block.num_hashed_tokens > evicted_block.num_hashed_tokens:
                    evicted_block = block
            # if evicted_block.num_hashed_tokens < block.num_hashed_tokens:
            #     evicted_block = block

        self.free_table.pop(evicted_block.block_hash)

        evicted_block.computed = False
        self.has_evicted = True
        # evicted_block.ref_seqs.clear()
        
        # logger.info(f'Block {evicted_block.block_number} (nht={evicted_block.num_hashed_tokens}) is evicted.')
        return evicted_block

    def add(self, block: PhysicalTokenBlock):
        self.free_table[block.block_hash] = block

    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        if block_hash not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block: PhysicalTokenBlock = self.free_table[block_hash]
        self.free_table.pop(block_hash)
        return block

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

class LRU2Evictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the PhysicalTokenBlock. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    def __init__(self):
        logger.info(f"{self.__class__.__name__} is used.")
        self.free_table: OrderedDict[int, PhysicalTokenBlock] = OrderedDict()

    def __contains__(self, block_hash: int) -> bool:
        return block_hash in self.free_table

    def evict(self) -> PhysicalTokenBlock:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        evicted_block = next(iter(self.free_table.values()))
        # The blocks with the lowest timestamps should be placed consecutively
        # at the start of OrderedDict. Loop through all these blocks to
        # find the one with maximum number of hashed tokens.
        for _, block in self.free_table.items():
            if evicted_block.last_accessed < block.last_accessed:
                break
            if evicted_block.num_hashed_tokens < block.num_hashed_tokens:
                evicted_block = block

        self.free_table.pop(evicted_block.block_hash)

        evicted_block.computed = False
        evicted_block.used_by_decode = False
        
        # evicted_block.ref_seqs.clear()
        return evicted_block

    def add(self, block: PhysicalTokenBlock):
        self.free_table[block.block_hash] = block

    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        if block_hash not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block: PhysicalTokenBlock = self.free_table[block_hash]
        self.free_table.pop(block_hash)
        return block

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    elif eviction_policy == EvictionPolicy.LRU2:
        return LRU2Evictor()
    elif eviction_policy == EvictionPolicy.LRU3:
        return LRU3Evictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
