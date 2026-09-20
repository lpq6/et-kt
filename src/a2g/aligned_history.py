"""Local candidate: include the latest observed event in shifted-history statistics."""

from .candidate import A2GModal


class A2GAlignedHistory(A2GModal):
    CANDIDATE_ID = "a2g_v44_aligned_history_local"

    def __init__(self, *args, use_aligned_history_statistics=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_aligned_history_statistics = bool(use_aligned_history_statistics)

    @property
    def history_stat_diagonal(self):
        # hist[:, t] is the observation at t-1, not the current target response.
        return 0 if self.use_aligned_history_statistics else -1
