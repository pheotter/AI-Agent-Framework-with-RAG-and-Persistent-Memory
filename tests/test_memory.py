import pytest
from unittest.mock import patch
import redis

from app.core.memory import RedisMemoryStore


@pytest.fixture
def mock_redis_client():
    """Create a fake Redis client so tests never touch a real Redis server."""
    with patch("redis.Redis") as mock:
        # `mock` is the patched Redis class.
        # `mock.return_value` is the fake client instance produced by redis.Redis(...).
        mock_instance = mock.return_value

        # Configure the client methods our tests depend on.
        mock_instance.ping.return_value = True
        mock_instance.delete.return_value = True
        mock_instance.keys.return_value = ["chat_history:session1", "chat_history:session2"]

        # Hand the configured fake client to the test.
        yield mock_instance


@pytest.fixture
def memory_store(mock_redis_client):
    """Create a RedisMemoryStore instance backed by the fake Redis client."""
    with patch("redis.Redis", return_value=mock_redis_client):
        # When RedisMemoryStore() calls redis.Redis(...), it receives
        # `mock_redis_client` instead of a real Redis connection.
        return RedisMemoryStore()


@pytest.fixture
def mock_redis_chat_history():
    """Create a fake LangChain Redis chat history object."""
    with patch("app.core.memory.RedisChatMessageHistory") as mock:
        yield mock.return_value


@pytest.fixture
def mock_conversation_buffer_memory():
    """Create a fake LangChain ConversationBufferMemory object."""
    with patch("app.core.memory.ConversationBufferMemory") as mock:
        yield mock.return_value


class TestRedisMemoryStore:
    """Tests for the Redis-backed conversation memory wrapper."""

    def test_init(self, mock_redis_client):
        """The constructor should create a Redis client and ping it once."""
        # We instantiate the class for the side effects in __init__.
        # The object itself is not needed later in this test.
        RedisMemoryStore()

        # If initialization succeeded, it must have tested the Redis connection.
        mock_redis_client.ping.assert_called_once()

    def test_init_connection_error(self):
        """Connection failures should be re-raised during initialization."""
        # Build a fresh patch here because this test needs a different behavior:
        # ping() should raise instead of returning True.
        with patch("redis.Redis") as mock:
            mock_instance = mock.return_value
            mock_instance.ping.side_effect = redis.ConnectionError("Test error")

            # The exception happens while RedisMemoryStore() runs __init__(),
            # so the constructor call itself must be inside pytest.raises(...).
            with pytest.raises(redis.ConnectionError):
                RedisMemoryStore()

    def test_get_history(self, memory_store):
        """get_history should wire RedisChatMessageHistory into LangChain memory."""
        with patch("app.core.memory.RedisChatMessageHistory") as mock_history_cls, patch(
            "app.core.memory.ConversationBufferMemory"
        ) as mock_memory_cls:
            mock_history = mock_history_cls.return_value
            mock_memory = mock_memory_cls.return_value

            result = memory_store.get_history("test_session")

            # The method should return the ConversationBufferMemory instance it builds.
            assert result == mock_memory

            # Verify the Redis-backed chat history constructor received the right session.
            mock_history_cls.assert_called_once()
            _, history_kwargs = mock_history_cls.call_args
            assert history_kwargs["session_id"] == "test_session"
            assert history_kwargs["key_prefix"] == "chat_history:"

            # Verify the higher-level memory object wraps that history correctly.
            mock_memory_cls.assert_called_once_with(
                memory_key="history",
                input_key="input",
                chat_memory=mock_history,
                return_messages=True,
            )

    def test_clear_history(self, memory_store):
        """clear_history should delete the Redis key for the session."""
        memory_store.clear_history("test_session")

        memory_store.redis_client.delete.assert_called_once_with("chat_history:test_session")

    def test_get_all_sessions(self, memory_store):
        """get_all_sessions should strip the shared key prefix from every session key."""
        sessions = memory_store.get_all_sessions()

        memory_store.redis_client.keys.assert_called_once_with("chat_history:*")
        assert sessions == ["session1", "session2"]

    def test_add_interaction(self, memory_store):
        """add_interaction should manually append one user and one AI message."""
        with patch("app.core.memory.RedisChatMessageHistory") as mock_history_cls:
            mock_history = mock_history_cls.return_value

            memory_store.add_interaction("test_session", "Hello", "Hi there")

            mock_history_cls.assert_called_once()
            _, history_kwargs = mock_history_cls.call_args
            assert history_kwargs["session_id"] == "test_session"
            assert history_kwargs["key_prefix"] == "chat_history:"
            mock_history.add_user_message.assert_called_once_with("Hello")
            mock_history.add_ai_message.assert_called_once_with("Hi there")
