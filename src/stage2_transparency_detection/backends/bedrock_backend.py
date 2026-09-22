import boto3
from botocore.exceptions import ClientError
import traceback

class BedrockBackend:
    """
    Wrapper around AWS Bedrock 'converse' API.

    Expects a BedrockConfig with:
      - region      (e.g. "us-east-1")
      - model_id    (e.g. "anthropic.claude-3-haiku-20240307-v1:0")
    """

    def __init__(self, bedrock_config, max_completion_tokens: int = 80):
        if not bedrock_config.region or not bedrock_config.model_id:
            raise ValueError(
                "Missing Bedrock region or model_id. "
                "Set them in config.py under BedrockConfig."
            )

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=bedrock_config.region,
        )
        self.model_id = bedrock_config.model_id
        self.max_completion_tokens = max_completion_tokens

    def classify(self, prompt: str) -> str | None:
        """
        Sends a user prompt to the Bedrock model and returns the raw text response.

        Uses `converse`:
        messages=[{"role": "user", "content": [{"text": prompt}]}]
        """
        conversation = [{"role": "user", "content": [{"text": prompt}]}]

        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=conversation,
                inferenceConfig={
                    "maxTokens": self.max_completion_tokens,
                    "temperature": 0.0,
                },
            )

            out = ""
            for item in response["output"]["message"]["content"]:
                if "text" in item:
                    out += item["text"]
            return out.strip()

        except ClientError as e:
            err = e.response.get("Error", {})
            print("Bedrock ClientError:")
            print("Code:", err.get("Code"))
            print("Message:", err.get("Message"))
            traceback.print_exc()
            return None

        except Exception as e:
            print("Bedrock error:", repr(e))
            traceback.print_exc()
            return None