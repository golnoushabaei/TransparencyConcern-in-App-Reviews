# src/stage2_transparency_detection/backends/openai_backend.py
from openai import OpenAI
import os

class OpenAIBackend:
    """
    Wrapper around OpenAI's chat completions API.

    Expects an OpenAIConfig with:
      - api_key
      - model_name
    """

    def __init__(self, openai_config, max_completion_tokens: int = 80):
        api_key = openai_config.api_key or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "Missing OpenAI API key."
                "Set it in config.py or via OPENAI_API_KEY."
            )

        self.client = OpenAI(api_key=api_key)
        self.model_name = openai_config.model_name
        self.max_completion_tokens = max_completion_tokens

    def classify(self, prompt: str) -> str:
        """
        Sending a user prompt to the OpenAI model and returning the raw text response.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.max_completion_tokens,
        )
        return response.choices[0].message.content.strip()