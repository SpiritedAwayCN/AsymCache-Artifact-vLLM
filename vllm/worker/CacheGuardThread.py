import threading
import time
import torch
from typing import Union, List
import vllm.distributed.kv_transfer.vllm_adapter as dist_kv
from vllm.core.block_manager_v1 import BlockSpaceManagerV1
from vllm.core.block_manager_v2 import BlockSpaceManagerV2
from vllm.distributed.kv_transfer.kv_pipe.torch_distributed_pipe import (
    TorchDistributedPipe)
from vllm.logger import init_logger
logger = init_logger(__name__)
class CacheGuardThread:
    def __init__(self, kv_agent: dist_kv.KV_transfer_agent, get_block_manager : Union[BlockSpaceManagerV1, BlockSpaceManagerV2], model_executable: torch.nn.Module, kv_caches: List[torch.Tensor]):
        self.kv_agent = kv_agent  # KV transfer 代理
        self.cache = kv_caches
        self.model_executable : torch.nn.Module = model_executable  # 模型层对象
        self.get_block_manager: Union[BlockSpaceManagerV1, BlockSpaceManagerV2] = get_block_manager  # 动态获取 block_manager 的引用
        self.running = True  # 控制守护线程的状态
        self.lock = threading.Lock()  # 加锁保证缓存访问安全
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        """启动守护线程"""
        logger.debug("Starting CacheGuardThread")
        self.thread.start()

    def stop(self):
        """停止守护线程"""
        logger.debug("Shutting down CacheGuardThread")
        self.running = False
        self.thread.join()

    def run(self):
        """守护线程主循环"""
        while self.running:
            try:
                # 使用 kv_agent 接收 hash 请求
                logger.debug("Receiving hash request")
                hash_request, full_hash = self.kv_agent.recv_hash()
                if hash_request is not None:
                    self.process_request(hash_request, full_hash)
            except Exception as e:
                print(f"CacheGuardThread error: {e}")
            time.sleep(0.01)

    def process_request(self, query_hashes, full_hashes):
        """检查 block manager 缓存是否满足需求，并直接发送 KV 缓存"""
        block_manager = self.get_block_manager
        assert block_manager is not None, "CRITICAL ERROR: block_manager is None"

        # 等待直到所有需要的 blocks 都满足条件
        all_blocks_ready = False
        while not all_blocks_ready:
            all_blocks_ready = block_manager.gpu_allocator.has_full_block(query_hashes)  # 假设初始所有 blocks 满足
            if not all_blocks_ready:
                time.sleep(0.001)  # 等待 10ms 再次检查

        # 满配后发送 KV 缓存
        logger.debug("sending KV")
        self.kv_agent.direct_send_kv_caches(
            model_executable=self.model_executable,  # 若需要模型层信息，可传入实际对象
            kv_caches=self.cache,
            query_hashes=query_hashes,
            full_hashes=full_hashes,
            block_manager=block_manager
        )
        logger.debug("KV sent")
