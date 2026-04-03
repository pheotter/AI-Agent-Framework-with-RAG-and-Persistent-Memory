import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from app.main import app
from app.services.llm_service import LLMService


@pytest.fixture
def test_client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_ai_agent():
    """Create a mock AI agent for testing."""
    with patch("app.core.agent.AIAgent") as mock:
        mock_instance = mock.return_value
        mock_instance.process_message = AsyncMock(
            return_value=(
                "Test response",
                [{"title": "Source", "url": "http://example.com"}],
            )
        )
        mock_instance.memory_store = MagicMock()
        mock_instance.memory_store.clear_history = MagicMock()
        yield mock_instance


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service for testing."""
    with patch("app.services.llm_service.get_llm_service") as mock:
        mock.return_value = MagicMock(spec=LLMService)
        yield mock.return_value


class TestChatEndpoints:
    """Test suite for chat endpoints."""

    @patch("app.routes.chat.AIAgent")
    async def test_chat_endpoint(
        self, mock_agent_class, test_client, mock_ai_agent, mock_llm_service
    ):
        mock_agent_class.return_value = mock_ai_agent

        response = test_client.post(
            "/chat/",
            json={"message": "Hello, AI", "session_id": "test_session"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Test response"
        assert len(data["sources"]) == 1
        assert data["session_id"] == "test_session"

        mock_ai_agent.process_message.assert_called_once_with(
            message="Hello, AI", session_id="test_session"
        )

    @patch("app.routes.chat.AIAgent")
    async def test_clear_chat_history_endpoint(
        self, mock_agent_class, test_client, mock_ai_agent
    ):
        mock_agent_class.return_value = mock_ai_agent

        response = test_client.delete("/chat/test_session")

        assert response.status_code == 200
        assert "message" in response.json()
        mock_ai_agent.memory_store.clear_history.assert_called_once_with("test_session")
