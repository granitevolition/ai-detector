# Predibase API Configuration
API_TOKEN = "pb_mXNhbmtfU-yyIhBhRow4pw"  # Your Predibase API token

# Project Configuration
PROJECT_NAME = "ai_detector"
MODEL_NAME = "ai_content_detector"

# Training Configuration
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  # You can change this to any supported model
ADAPTER_TYPE = "lora"  # Options: "lora", "turbo_lora", "turbo"
EPOCHS = 4
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
MAX_SEQ_LENGTH = 2048
GRADIENT_ACCUMULATION_STEPS = 8

# Binary classification settings
# For our AI detection model, we're using a binary classification approach
HUMAN_LABEL = 0  # Represents human-written text
AI_LABEL = 1     # Represents AI-generated text

# Deployment Configuration
DEPLOYMENT_TYPE = "shared"  # Options: "shared", "dedicated"
INSTANCE_TYPE = "gpu.t4.1x"  # For dedicated deployments
MIN_REPLICAS = 1
MAX_REPLICAS = 1

# Inference Configuration
TEMPERATURE = 0.0  # Lower temperature for more deterministic outputs
MAX_TOKENS = 0    # Maximum length of generated response