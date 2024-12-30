from collections import OrderedDict
from vllm.block import PhysicalTokenBlock
from vllm.core.evictor_v1 import Evictor

class BufferedEvictor:
    """
    BufferedEvictor manages a buffer without any evictor.
    """

    def __init__(self, evictor: Evictor):
        self.buffer: OrderedDict[int, PhysicalTokenBlock] = OrderedDict()
        self.evictor = evictor

    def add(self, block: PhysicalTokenBlock):
        """
        Adds a block to the buffer.
        """
        block.computed = True
        assert block.used_by_decode == False
        self.buffer[block.block_hash] = block

    def pop(self, block_hash: int) -> PhysicalTokenBlock:
        """
        Removes the block with the given hash from the buffer
        and places it into the evictor.
        """
        if block_hash not in self.buffer:
            # print(f"Block with hash {block_hash} not in buffer with buffer size {len(self.buffer)}")
            return None
        else:
            # print(f"pop block {block_hash}")
            block = self.buffer.pop(block_hash)
            # print(f"block used by decode: {block.used_by_decode}")
            assert block is not None
            block.used_by_decode == True
            self.evictor.add(block)
            return block

    def get(self, block_hash: int) -> PhysicalTokenBlock:
        """
        Returns the block with the given hash from the buffer.
        """
        return self.buffer.pop(block_hash, None)
    
    def show_size(self) -> int:
        """
        Returns the current size of the buffer.
        """
        return len(self.buffer)
    
    def flush(self, hash_list: list[int]):
        """
        Transfers all blocks in the buffer with hashes in hash_list to the evictor.
        """
        print(self.show_size())
        for block_hash in hash_list:
            if block_hash in self.buffer:
                print(f"flushing block {block_hash}")
                self.pop(block_hash)
        print(self.show_size())

    def check_if_in_buffer(self, block_hash: int) -> bool:
        """
        Returns True if the block with the given hash is in the buffer.
        """
        # check if block is computed
        if block_hash in self.buffer:
            assert self.buffer[block_hash].computed == True
            return True
        else:
            return False
    
    def show(self):
        """
        Prints the hashes of the blocks in the buffer.and the block'used by decode
        """
        print("BufferedEvictor:")
        for block_hash, block in self.buffer.items():
            print(f"Block {block_hash} used by decode: {block.used_by_decode}")
        print("End of BufferedEvictor")