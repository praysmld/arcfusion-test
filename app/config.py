import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # API Configuration
    app_name: str = "Chat With PDF Backend"
    version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # LLM Configuration
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = "gpt-4-turbo-preview"
    
    # Web Search Configuration
    tavily_api_key: Optional[str] = os.getenv("TAVILY_API_KEY")
    
    # Vector Database Configuration
    chroma_persist_directory: str = "./data/chroma"
    embedding_model: str = "text-embedding-3-small"
    
    # Session Management
    session_ttl: int = 3600  # 1 hour
    max_session_memory: int = 50  # Maximum messages per session
    
    # PDF Processing
    pdf_directory: str = "./papers"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Agent Configuration
    max_iterations: int = 10
    agent_timeout: int = 60
    
    class Config:
        env_file = ".env"


settings = Settings() 