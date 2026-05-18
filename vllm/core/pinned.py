# SPDX-License-Identifier: Apache-2.0
from vllm.sequence import SequenceGroup


def calcTTL(seq_group: SequenceGroup) -> int:
    """Return the TTL (seconds) for a finished request.

    This is a placeholder implementation and always returns 0.
    """
    return 5
