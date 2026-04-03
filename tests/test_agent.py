import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.agent import AIAgent
from app.services.llm_service import LLMService


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service for testing."""
    service = MagicMock(spec=LLMService)
    mock_llm = AsyncMock()
    mock_llm.arun = AsyncMock(return_value="This is a test response")
    service.get_llm.return_value = mock_llm
    return service


@pytest.fixture
def mock_qdrant_service():
    """Create a mock Qdrant service for testing."""
    with patch("app.services.qdrant_service.QdrantService") as mock:
        mock_instance = mock.return_value
        mock_instance.search.return_value = [
            {
                "title": "Knowledge Base Entry",
                "content": "This is a test entry from the knowledge base.",
                "score": 0.85,
            }
        ]
        yield mock_instance


@pytest.fixture
def mock_memory_store():
    """Create a mock Redis memory store for testing."""
    with patch("app.core.memory.RedisMemoryStore") as mock:
        mock_instance = mock.return_value
        mock_instance.get_history.return_value = MagicMock()
        yield mock_instance


@pytest.fixture
def agent(mock_llm_service, mock_qdrant_service, mock_memory_store):
    """Create an AI agent with mock dependencies for testing."""
    with patch("app.core.agent.QdrantService", return_value=mock_qdrant_service), patch(
        "app.core.agent.RedisMemoryStore", return_value=mock_memory_store
    ):
        yield AIAgent(mock_llm_service)


class TestAIAgent:
    """Test suite for the AI Agent core functionality."""

    @pytest.mark.asyncio
    async def test_process_message(self, agent):
        response, sources = await agent.process_message(
            "What is artificial intelligence?", "test_session"
        )

        assert response == "This is a test response"
        assert len(sources) == 1
        agent.memory_store.get_history.assert_called_once_with("test_session")
        agent.qdrant_service.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        agent.llm.arun.side_effect = Exception("Test error")

        with pytest.raises(Exception):
            await agent.process_message("Test error message", "test_session")

    @pytest.mark.asyncio
    async def test_memory_interaction(self, agent):
        await agent.process_message("Test memory", "test_session")

        agent.memory_store.get_history.assert_called_once_with("test_session")
        agent.memory_store.add_interaction.assert_called_once()

    def test_reset(self, agent):
        agent.reset("test_session")

        agent.memory_store.clear_history.assert_called_once_with("test_session")
