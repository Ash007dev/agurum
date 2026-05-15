import os
import pytest
from engine.config import Config

def test_config_defaults():
    # Clear environment variables
    for k in ["ANTHROPIC_API_KEY", "UDS_PATH", "RING_BUFFER_CAP", "BATCH_SIZE", "BATCH_INTERVAL_MS", "THREAD_POOL_WORKERS"]:
        if k in os.environ:
            del os.environ[k]
    
    config = Config()
    assert config.ANTHROPIC_API_KEY == ""
    assert config.UDS_PATH == "/tmp/pce.sock"
    assert config.RING_BUFFER_CAP == 10000
    assert config.BATCH_SIZE == 100
    assert config.BATCH_INTERVAL_MS == 100
    assert config.THREAD_POOL_WORKERS == 4

def test_config_overrides():
    os.environ["ANTHROPIC_API_KEY"] = "sk-test"
    os.environ["THREAD_POOL_WORKERS"] = "8"
    config = Config()
    assert config.ANTHROPIC_API_KEY == "sk-test"
    assert config.THREAD_POOL_WORKERS == 8
