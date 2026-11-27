"""Learned-solver support in core (numpy only, no torch): observer embedding and model inputs (SPEC 5.6)."""
from .embed import Z_DIM, embed, embed_observer, embed_profile, embed_samples, embed_many, proxy_matrix  # noqa: F401
from .features import F_DIM, scene_inputs, group_arrays, cat_rgb_matrix  # noqa: F401
