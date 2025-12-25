"""
Local Qwen 2.5 3B LLM Client
Uses HuggingFace Transformers for local inference
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

class LocalLLMClient:
    """Singleton client for local Qwen 2.5 3B model"""

    _instance: Optional['LocalLLMClient'] = None
    _model = None
    _tokenizer = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the model only once"""
        if self._model is None:
            self._initialize_model()

    def _initialize_model(self):
        """Load the Qwen 2.5 3B model"""
        model_name = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")

        print(f"Loading model: {model_name}")
        print("This may take a moment on first run...")

        # Determine device
        if torch.cuda.is_available():
            self._device = "cuda"
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self._device = "cpu"
            print("Using CPU (inference will be slower)")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load model with appropriate dtype
        if self._device == "cuda":
            # Use float16 for GPU to save memory with better memory management
            # Limit GPU usage to avoid OOM on 6GB cards
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "4.5GB", "cpu": "8GB"}  # Reserve GPU memory safely
            )
        else:
            # Use float32 for CPU
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True
            )

        self._model.eval()
        print(f"Model loaded successfully on {self._device}")

    def generate(self, system_prompt: str, user_content: str, max_new_tokens: int = 256) -> str:
        """
        Generate response using the local model

        Args:
            system_prompt: System instruction for the model
            user_content: User query/content
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        try:
            # Format the conversation using Qwen's chat template
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            # Apply chat template
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            model_inputs = self._tokenizer([text], return_tensors="pt").to(self._device)

            # Generate
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )

            # Decode only the new tokens (remove input)
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip()

        except Exception as e:
            return f"ERROR: Failed to generate response - {str(e)}"


# Create singleton instance
_llm_client = None

def get_llm_client() -> LocalLLMClient:
    """Get or create the LLM client singleton"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LocalLLMClient()
    return _llm_client


def call_local_llm(system_prompt: str, user_content: str, max_new_tokens: int = 256) -> str:
    """
    Call the local LLM - compatible interface with the old call_deepseek function

    Args:
        system_prompt: System instruction
        user_content: User query
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated response text
    """
    client = get_llm_client()
    return client.generate(system_prompt, user_content, max_new_tokens)
