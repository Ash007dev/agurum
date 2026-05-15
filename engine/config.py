import os
from dataclasses import dataclass, field

@dataclass
class Config:
    ANTHROPIC_API_KEY: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    UDS_PATH: str = field(default_factory=lambda: os.getenv("UDS_PATH", "/tmp/pce.sock"))
    RING_BUFFER_CAP: int = field(default_factory=lambda: int(os.getenv("RING_BUFFER_CAP", "10000")))
    BATCH_SIZE: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "100")))
    BATCH_INTERVAL_MS: int = field(default_factory=lambda: int(os.getenv("BATCH_INTERVAL_MS", "100")))
    THREAD_POOL_WORKERS: int = field(default_factory=lambda: int(os.getenv("THREAD_POOL_WORKERS", "4")))

config = Config()
