from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkingConfig(BaseModel):
    """Validated V2 limits. Token limits apply to enriched embedding text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_root: Path = Path("data/processed/cleaned_documents")
    output_root: Path = Path("data/processed/v2")
    min_child_tokens: int = Field(default=60, ge=1)
    preferred_child_min: int = Field(default=180, ge=1)
    preferred_child_max: int = Field(default=320, ge=1)
    max_child_tokens: int = Field(default=450, ge=1)
    preferred_parent_min: int = Field(default=600, ge=1)
    preferred_parent_max: int = Field(default=1200, ge=1)
    max_parent_tokens: int = Field(default=1500, ge=1)

    @model_validator(mode="after")
    def ordered_limits(self) -> "ChunkingConfig":
        if not (
            self.min_child_tokens
            <= self.preferred_child_min
            <= self.preferred_child_max
            <= self.max_child_tokens
        ):
            raise ValueError("child token limits must be monotonically increasing")
        if not (
            self.preferred_parent_min
            <= self.preferred_parent_max
            <= self.max_parent_tokens
        ):
            raise ValueError("parent token limits must be monotonically increasing")
        return self
