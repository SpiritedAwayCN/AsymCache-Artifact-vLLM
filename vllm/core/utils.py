from typing import Dict, List, Optional
from typing import Iterable, Tuple, List

from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device

BlockId = int

class RefSeqTracker:
    def __init__(self):
        self.phid_to_seqs : Dict[Device, Dict[BlockId, List[Sequence]]]= {
            Device.CPU : {},
            Device.GPU : {}
        }

    def add(self, block_id: BlockId, seq: Sequence):
        dct = self.phid_to_seqs[seq.device]
        if block_id not in dct:
            dct[block_id] = [seq]
        else:
            dct[block_id].append(seq)
    
    def add_all(self, block_ids: Iterable[BlockId], seq: Sequence):
        dct = self.phid_to_seqs[seq.device]
        for block_id in block_ids:
            if block_id not in dct:
                dct[block_id] = [seq]
            else:
                dct[block_id].append(seq)

    def remove(self, 
               seq: Sequence, 
               block_ids: Optional[Iterable[BlockId]] = None,
               device: Optional[Device] = None):
        if block_ids is None:
            block_ids = self.phid_to_seqs.keys()
        devices = [device] if device is not None else self.phid_to_seqs.keys()
        for device in devices:
            for block_id in block_ids:
                if block_id in self.phid_to_seqs:
                    self.phid_to_seqs[device][block_id].remove(seq)
    
    def get(self, block_id: BlockId, device: Device = Device.GPU) -> List[Sequence]:
        return self.phid_to_seqs[device][block_id]
    
    def reset(self, device: Optional[Device] = None):
        if device is not None:
            self.phid_to_seqs[device] = {}
        else:
            self.phid_to_seqs = {
                Device.CPU : {},
                Device.GPU : {}
            }

    def __repr__(self):
        return f"RefSeqTracker({self.phid_to_seqs!r})"