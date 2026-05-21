# SPDX-License-Identifier: Apache-2.0
from vllm.sequence import SequenceGroup


def calcTTL(seq_group: SequenceGroup) -> int:
    """Return the TTL (seconds) for a finished request."""
    if seq_group.program_id is None or seq_group.estimated_ttl <= 0:
        return 0
    
    return seq_group.estimated_ttl
