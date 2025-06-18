import threading
import time
import torch
import traceback
from typing import Union, List, Tuple
import vllm.distributed.kv_transfer.vllm_adapter as dist_kv
from vllm.core.block_manager_v1 import BlockSpaceManagerV1
from vllm.core.block_manager import BlockSpaceManager
from vllm.distributed.kv_transfer.kv_pipe.torch_distributed_pipe import (
    TorchDistributedPipe)
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.worker.CacheGuardLayerState import CacheGuardLayerState
logger = init_logger(__name__)
class CacheGuardThread:
    def __init__(self, 
                 kv_agent: dist_kv.KV_transfer_agent, 
                 get_block_manager : Union[BlockSpaceManagerV1, BlockSpaceManager], 
                 model_executable: torch.nn.Module, 
                 kv_caches: List[torch.Tensor],
                 is_driver: bool = True,
                 guard_layer_state: CacheGuardLayerState = None):
        self.kv_agent = kv_agent  # KV transfer 代理
        self.cache = kv_caches
        self.model_executable : torch.nn.Module = model_executable  # 模型层对象
        self.get_block_manager: Union[BlockSpaceManagerV1, BlockSpaceManager] = get_block_manager  # 动态获取 block_manager 的引用
        self.running = True  # 控制守护线程的状态
        self.lock = threading.Lock()  # 加锁保证缓存访问安全
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.is_driver = is_driver

        self.guard_layer_state = guard_layer_state

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
        try:
            if self.is_driver:
                while self.running:
                    try:
                        # 使用 kv_agent 接收 hash 请求
                        logger.debug("Receving hash request (driver)")
                        hash_request, full_hash = self.kv_agent.recv_hash()
                        logger.debug("Recevied hash request (driver)")
                        if hash_request is not None:
                            self.process_request(hash_request, full_hash)
                    except Exception as e:
                        print(f"CacheGuardThread error (driver): {e}")
                        traceback.print_exc()
                    time.sleep(0.01)
            else:
                while self.running:
                    try:
                        logger.debug("Receving dummy hash request (non-driver)")
                        self.kv_agent.recv_hash_dummy()
                        logger.debug("Recevied dummy hash request (non-driver)")
                        self.process_request(None, None)
                    except Exception as e:
                        print(f"CacheGuardThread error (non-driver): {e}")
                        traceback.print_exc()
                    time.sleep(0.01)
        except Exception as e:
            print(f"CacheGuardThread error: {e}")
            traceback.print_exc()

    def process_request(self, query_hashes, full_hashes):
        """检查 block manager 缓存是否满足需求，并直接发送 KV 缓存"""
        block_manager = self.get_block_manager
        assert (not self.is_driver) or (block_manager is not None), "CRITICAL ERROR: block_manager is None"

        if self.is_driver:
            # 等待直到所有需要的 blocks 都满足条件
            # all_blocks_ready = False
            # while True:
            #     all_blocks_ready = block_manager.gpu_allocator.has_full_block(query_hashes)  # 假设初始所有 blocks 满足
            #     if all_blocks_ready:
            #         break
            #     self.get_block_manager.gpu_allocator.cacheguard_event.wait()  # 等待缓存满足条件
                
            # Function to retrieve block IDs and matching hashes based on query hashes
            def get_block_ids(query_hashes: List[int], full_hashes: List[int], block_manager: Union[BlockSpaceManagerV1, BlockSpaceManagerV2]) -> Tuple[List[int], List[int]]:
                blockid_list = []
                hash_match_list = []
                logger.debug(f"getting blockids")
                for hash in query_hashes:
                    # logger.debug(f"getting blockids:{hash}")
                    block_ids = block_manager.get_hash_block_ids(hash)
                    if block_ids:
                        blockid_list.append(block_ids)
                        hash_match_list.append(hash)
                print(f"blockid_list: {blockid_list}")
                block_manager.gpu_allocator.change_block_used_status(full_hashes, True)
                block_manager.gpu_allocator.show_evictor_buffer()
                
                return blockid_list, hash_match_list

            block_ids, hash_matches = get_block_ids(query_hashes, full_hashes, block_manager)

            logger.debug("Sending broadcast data (block_ids)")
            get_tp_group().broadcast_object_secondary(
                {"block_ids": block_ids, "hash_matches": hash_matches},
                src=0
            )
            logger.debug("Sent broadcast data (block_ids)")
        else:
            logger.debug("Waiting broadcast data (block_ids)")
            data = get_tp_group().broadcast_object_secondary({}, src=0)
            logger.debug(f"Received broadcast data, keys: {data.keys()}")
            block_ids = data["block_ids"]
            hash_matches = data["hash_matches"]

        # 满配后发送 KV 缓存
        logger.debug("sending KV")
        self.kv_agent.direct_send_kv_caches(
            model_executable=self.model_executable,  # 若需要模型层信息，可传入实际对象
            kv_caches=self.cache,
            block_ids=block_ids,
            hash_matches=hash_matches,
            is_driver=self.is_driver,
            guard_layer_state=self.guard_layer_state,
            instance_group=get_tp_group(),
        )
        logger.debug("KV sent")
