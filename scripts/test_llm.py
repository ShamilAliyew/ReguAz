from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.reguaz.services.generation.llm_factory import LLMFactory

llm = LLMFactory.create()

response = llm.generate(
    """
You are a helpful AI assistant.

Question:
Alma meyvesi haqqinda azerbaycan dilinde melumat ver.

Answer:
"""
)

print(response)