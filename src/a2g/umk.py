"""Development-only v45 identity; all UMK switches default to disabled."""

from .candidate import A2GModal


class A2GUMK(A2GModal):
    CANDIDATE_ID = "a2g_v45_umk_local_candidate"
