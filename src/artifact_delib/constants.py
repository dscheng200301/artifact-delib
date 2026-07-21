"""Constants for the ArtifactDelib framework."""

# Default VLM model
DEFAULT_MODEL = "qwen3.5-flash-2026-02-23"

# JSON response schema for API calls
JSON_RESPONSE_SCHEMA = {"type": "json_object"}

# Route action types
ROUTE_FAST = "FAST"
ROUTE_SHAPE_RECHECK = "SHAPE_RECHECK"
ROUTE_STYLE_RECHECK = "STYLE_RECHECK"
ROUTE_GLYPH_RECHECK = "GLYPH_RECHECK"
ROUTE_MATERIAL_RECHECK = "MATERIAL_RECHECK"
ROUTE_LOCAL_DETAIL_RECHECK = "LOCAL_DETAIL_RECHECK"
ROUTE_DELIBERATION = "DELIBERATION"

# Expert names
EXPERT_SHAPE = "shape"
EXPERT_STYLE = "style"
EXPERT_GLYPH = "glyph"
EXPERT_MATERIAL = "material"
EXPERT_LOCAL_DETAIL = "local_detail"

# Default K for candidates
DEFAULT_TOP_K = 3

# Default deliberation limits
MAX_DELIBERATION_ROUNDS = 2
MAX_RECHECK_ROUNDS = 2
