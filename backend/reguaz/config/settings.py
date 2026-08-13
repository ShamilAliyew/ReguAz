import os
from pathlib import Path

from dotenv import load_dotenv

# =============================================================================
# Project Paths
# =============================================================================

# This resolves to the directory containing this file: backend/reguaz/config
_CURRENT_DIR = Path(__file__).resolve().parent

# The backend root is 2 levels up: backend/reguaz/config -> backend/reguaz -> backend
BACKEND_ROOT = _CURRENT_DIR.parents[1]
REPOSITORY_ROOT = BACKEND_ROOT.parent

# Resolve .env from repository location, independent of process working directory.
load_dotenv(REPOSITORY_ROOT / ".env")


def _repository_path(value: str, repository_root: Path) -> Path:
    """Resolve relative configuration paths against repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repository_root / path).resolve()


# Default local paths
_DEFAULT_DATA_DIR = BACKEND_ROOT.parent / "data"
if not _DEFAULT_DATA_DIR.exists():
    _DEFAULT_DATA_DIR = BACKEND_ROOT / "data"

# Default Qdrant path is inside backend folder now: backend/qdrant_data/
_DEFAULT_QDRANT_PATH = BACKEND_ROOT / "qdrant_data"
if not _DEFAULT_QDRANT_PATH.exists():
    # Fallback to repo data/qdrant if not created yet
    _DEFAULT_QDRANT_PATH = _DEFAULT_DATA_DIR / "qdrant"

_DEFAULT_LLM_MODEL_PATH = (
    BACKEND_ROOT / "reguaz" / "models" / "gemma-4-E4B-it-Q4_K_M.gguf"
)
_DEFAULT_GOLD_DATASET_PATH = (
    _DEFAULT_DATA_DIR / "evaluation" / "gold_dataset_for_llm_generation.xlsx"
)
_DEFAULT_RESULTS_PATH = BACKEND_ROOT / "results"
_DEFAULT_LOGS_PATH = BACKEND_ROOT / "logs"

# Use BACKEND_ROOT as PROJECT_ROOT to ensure standalone backend runs completely within the backend directory
PROJECT_ROOT = BACKEND_ROOT

# Environment overridden paths
QDRANT_PATH = _repository_path(
    os.getenv("QDRANT_PATH", str(_DEFAULT_QDRANT_PATH)), REPOSITORY_ROOT
)
LLM_MODEL_PATH = _repository_path(
    os.getenv("LLM_MODEL_PATH", str(_DEFAULT_LLM_MODEL_PATH)), REPOSITORY_ROOT
)
GOLD_DATASET_PATH = _repository_path(
    os.getenv("GOLD_DATASET_PATH", str(_DEFAULT_GOLD_DATASET_PATH)), REPOSITORY_ROOT
)
RESULTS_PATH = _repository_path(
    os.getenv("RESULTS_PATH", str(_DEFAULT_RESULTS_PATH)), REPOSITORY_ROOT
)
LOGS_PATH = _repository_path(
    os.getenv("LOGS_PATH", str(_DEFAULT_LOGS_PATH)), REPOSITORY_ROOT
)

# Sub-directories
DATA_DIR = _DEFAULT_DATA_DIR
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
METADATA_DIR = PROCESSED_DIR / "metadata"

# Create required directories automatically
LOGS_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
QDRANT_PATH.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Qdrant
# =============================================================================
DEFAULT_COLLECTION_PREFIX = "reguaz"
DEFAULT_DISTANCE_METRIC = "Cosine"

# =============================================================================
# Embeddings
# =============================================================================
DEFAULT_BATCH_SIZE = 100
SUPPORTED_EMBEDDING_MODELS = (
    "bge_m3",
    "e5",
)

# =============================================================================
# Retrieval
# =============================================================================
DEFAULT_TOP_K = 10
BM25_TOP_K = 15
SEMANTIC_TOP_K = 15
RRF_K = 60

# =============================================================================
# Evaluation
# =============================================================================
DEFAULT_EVALUATION_TOP_K = 5

# =============================================================================
# Logging
# =============================================================================
DEFAULT_LOG_LEVEL = "INFO"

# =============================================================================
# LLM Generation
# =============================================================================
DEFAULT_LLM_TYPE = "gemma"

# Common LLM Parameters
LLM_CONTEXT_WINDOW = 8192
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 512
LLM_TOP_P = 0.95
LLM_TOP_K = 40
LLM_REPEAT_PENALTY = 1.1
LLM_SEED = 42
LLM_GPU_LAYERS = int(os.getenv("LLM_GPU_LAYERS", "-1"))
PROMPT_RESERVED_TOKENS = 300

# NVIDIA NIM GPT-OSS
NVIDIA_API_BASE = os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
NVIDIA_GPT_OSS_MODEL = os.getenv("NVIDIA_GPT_OSS_MODEL", "openai/gpt-oss-120b")
NVIDIA_GPT_OSS_CONTEXT_WINDOW = int(
    os.getenv("NVIDIA_GPT_OSS_CONTEXT_WINDOW", "131072")
)
NVIDIA_GPT_OSS_MAX_TOKENS = int(os.getenv("NVIDIA_GPT_OSS_MAX_TOKENS", "4096"))
NVIDIA_GPT_OSS_TIMEOUT_SECONDS = float(
    os.getenv("NVIDIA_GPT_OSS_TIMEOUT_SECONDS", "300")
)

# Groq GPT-OSS
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_GPT_OSS_20B_MODEL = os.getenv("GROQ_GPT_OSS_20B_MODEL", "openai/gpt-oss-20b")
GROQ_GPT_OSS_120B_MODEL = os.getenv("GROQ_GPT_OSS_120B_MODEL", "openai/gpt-oss-120b")
GROQ_GPT_OSS_CONTEXT_WINDOW = int(os.getenv("GROQ_GPT_OSS_CONTEXT_WINDOW", "131072"))
GROQ_GPT_OSS_MAX_TOKENS = int(os.getenv("GROQ_GPT_OSS_MAX_TOKENS", "2048"))
GROQ_GPT_OSS_TIMEOUT_SECONDS = float(os.getenv("GROQ_GPT_OSS_TIMEOUT_SECONDS", "90"))
GROQ_GPT_OSS_REASONING_EFFORT = os.getenv("GROQ_GPT_OSS_REASONING_EFFORT", "low")

# Model Paths
GEMMA_MODEL_PATH = BACKEND_ROOT / "reguaz" / "models" / "gemma-4-E4B-it-Q4_K_M.gguf"
