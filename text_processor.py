"""Text processor with optional LLM integration for text transformation."""

import json
from typing import Optional
import requests


class TextProcessor:
    """Process text with optional LLM transformation."""
    
    def __init__(
        self,
        backend: str = "none",
        model: str = "llama3.2",
        api_url: str = "http://localhost:11434/api/chat",
        timeout: int = 30
    ):
        """
        Initialize text processor.
        
        Args:
            backend: LLM backend ("ollama" or "none")
            model: Model name to use
            api_url: API endpoint URL
            timeout: Request timeout in seconds
        """
        self.backend = backend.lower()
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        
    def process(self, text: str, system_prompt: Optional[str] = None) -> str:
        """
        Process text, optionally using LLM transformation.
        
        Args:
            text: Input text to process
            system_prompt: Optional system prompt for LLM transformation
            
        Returns:
            Processed text
        """
        # If no system prompt or backend is "none", return original text
        if not system_prompt or self.backend == "none":
            return text
        
        # Process with LLM based on backend
        if self.backend == "ollama":
            return self._process_with_ollama(text, system_prompt)
        else:
            print(f"Warning: Unknown LLM backend '{self.backend}', using original text")
            return text
    
    def _process_with_ollama(self, text: str, system_prompt: str) -> str:
        """
        Process text using Ollama API.
        
        Args:
            text: Input text
            system_prompt: System prompt for transformation
            
        Returns:
            Transformed text
        """
        try:
            # Prepare the request
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                "stream": False
            }
            
            # Make the API request
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            
            # Check for errors
            if response.status_code != 200:
                print(f"Warning: Ollama API returned status {response.status_code}")
                print(f"Response: {response.text}")
                return text
            
            # Parse response
            result = response.json()
            
            # Extract the transformed text
            if "message" in result and "content" in result["message"]:
                transformed_text = result["message"]["content"].strip()
                print(f"[LLM] Transformed: '{text}' -> '{transformed_text}'")
                return transformed_text
            else:
                print(f"Warning: Unexpected response format from Ollama: {result}")
                return text
                
        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to connect to Ollama API: {e}")
            print("Using original text without LLM transformation")
            return text
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse Ollama response: {e}")
            return text
        except Exception as e:
            print(f"Warning: Unexpected error during LLM processing: {e}")
            return text


def create_processor_from_config(config: dict) -> TextProcessor:
    """
    Create a text processor from configuration dictionary.
    
    Args:
        config: Configuration dictionary with LLM settings
        
    Returns:
        Configured TextProcessor instance
    """
    llm_config = config.get("llm", {})
    
    return TextProcessor(
        backend=llm_config.get("backend", "none"),
        model=llm_config.get("model", "llama3.2"),
        api_url=llm_config.get("api_url", "http://localhost:11434/api/chat"),
        timeout=llm_config.get("timeout", 30)
    )
