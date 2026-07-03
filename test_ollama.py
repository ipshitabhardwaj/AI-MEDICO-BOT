"""
Test Ollama
"""

from utils.llm import LLMManager

llm = LLMManager(
    provider="Ollama",
    model_name="Llama 3.2"
).get_llm()

response = llm.invoke("Explain Machine Learning.")

print("=" * 60)
print("Ollama Response")
print("=" * 60)
print(response.content)