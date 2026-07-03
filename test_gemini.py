"""
Test Google Gemini
"""

from utils.llm import LLMManager

llm = LLMManager(
    provider="Google Gemini",
    model_name="Gemini 2.5 Flash"
).get_llm()

response = llm.invoke("What is Artificial Intelligence?")

print("=" * 60)
print("Google Gemini Response")
print("=" * 60)
print(response.content)