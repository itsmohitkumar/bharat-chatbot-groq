import os
import requests

class Config:
    API_KEY_ENV_VAR = 'GROQ_API_KEY'
    MODEL_OPTIONS_URL = "https://api.groq.com/openai/v1/models"
    DEFAULT_MODEL = "mixtral-8x7b-32768"

    @staticmethod
    def get_api_key():
        """Retrieve API key from environment variables."""
        api_key = os.getenv(Config.API_KEY_ENV_VAR)
        if not api_key:
            raise ValueError(f"API key not found. Please set the '{Config.API_KEY_ENV_VAR}' environment variable.")
        return api_key

    @staticmethod
    def get_model_options(api_key):
        """Fetch available model options from the Groq API."""
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        response = requests.get(Config.MODEL_OPTIONS_URL, headers=headers)
        response.raise_for_status()
        models = response.json()
        return {model['id'] for model in models['data']}

    @staticmethod
    def setup_langchain():
        """Setup LangChain environment variables and project."""
        langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
        if not langchain_api_key:
            raise ValueError("LANGCHAIN_API_KEY environment variable is not set.")
        
        # Set project name and environment variables
        project_name = "Bharat_Chatbot"
        os.environ["LANGCHAIN_PROJECT"] = project_name
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
