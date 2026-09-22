# src/stage2_transparency_detection/backends/azure_backend.py
from openai import AzureOpenAI
import os

class AzureBackend:
    """
    Wrapper around Azure OpenAI chat completions.

    Expects an AzureConfig with:
      - api_key
      - endpoint
      - api_version
      - model_name
    """

    def __init__(self, azure_config, max_completion_tokens: int = 80):
        # allowing env vars for secrets if config.py omits them
        api_key = azure_config.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = azure_config.endpoint or os.getenv("AZURE_OPENAI_API_ENDPOINT")
        api_version = azure_config.api_version

        if not api_key or not endpoint:
            raise ValueError(
                "Missing Azure API key or endpoint."
                "Set them in config.py or via AZURE_OPENAI_API_KEY / AZURE_OPENAI_API_ENDPOINT."
            )

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self.model_name = azure_config.model_name
        self.max_completion_tokens = max_completion_tokens

    def classify(self, prompt: str) -> str:
        """
        Sending a user prompt to the Azure model and returning the raw text response.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=self.max_completion_tokens,
        )
        return response.choices[0].message.content.strip()