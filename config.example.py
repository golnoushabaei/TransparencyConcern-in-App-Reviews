# config.example.py
from pathlib import Path
from dataclasses import dataclass

# files + experiment config

@dataclass
class FileConfig:
    """
    General file configuration for Stage 2 LLM experiments.
    """
    input_path: Path = Path("data/ground_truth/ground-truth-final.csv")
    output_path: Path = Path("results/llm_exp1_azure.csv")

    text_col: str = "Review"
    pred_col: str = "Pred Label"
    just_col: str = "LLM Justification"
    ground_truth_col: str = "Label"  # set "" if no ground truth available


@dataclass
class AzureConfig:
    """
    Azure OpenAI configuration.
    """
    api_key: str = "your-azure-api-key"
    endpoint: str = "https://your-azure-endpoint/"
    api_version: str = "your-api-version"
    model_name: str = "your-azure-model-name"  # e.g. "gpt-4o" or deployment name


@dataclass
class BedrockConfig:
    """
    AWS Bedrock configuration.
    """
    region: str = "bedrock-region"
    model_id: str = "bedrock-model-id"


@dataclass
class OpenAIConfig:
    """
    OpenAI (non-Azure) configuration.
    """
    api_key: str = "your-openai-api-key"
    model_name: str = "your-openai-model-name"  # e.g. "gpt-4o-mini"


@dataclass
class PromptConfig:
    """
    Prompt template selection for Stage 2 LLM experiments.
    Maps to files under `prompts/`.
    """
    # e.g. "llm-experiment-prompt1" -> prompts/llm-experiment-prompt1.txt
    template_name: str = "llm-experiment-prompt1"


@dataclass
class RunConfig:
    """
    Runtime options shared across scripts.
    """
    backend: str = "azure"           # "azure", "openai", "bedrock"
    sleep_seconds: float = 0.05      # delay between LLM calls for safety
    max_completion_tokens: int = 80  # default for LLM experiments

@dataclass
class Config:
    files: FileConfig = FileConfig()
    azure: AzureConfig = AzureConfig()
    bedrock: BedrockConfig = BedrockConfig()
    openai: OpenAIConfig = OpenAIConfig()
    prompt: PromptConfig = PromptConfig()
    run: RunConfig = RunConfig()

    @staticmethod
    def get_default_config() -> "Config":
        return Config()