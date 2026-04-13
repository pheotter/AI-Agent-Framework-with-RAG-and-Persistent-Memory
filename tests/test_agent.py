import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.agent import AIAgent
from app.services.llm_service import LLMService


@pytest.fixture
def mock_llm_service():
    """Create a fake LLM service that returns an async-capable mock LLM."""
    service = MagicMock(spec=LLMService)
    mock_llm = AsyncMock()
    mock_llm.arun = AsyncMock(return_value="This is a test response")
    service.get_llm.return_value = mock_llm
    return service


@pytest.fixture
def mock_qdrant_service():
    """Create a fake Qdrant service so retrieval stays local to the test."""
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
def mock_llm_chain():
    """Create a fake LLMChain with an async `arun` method."""
    with patch("app.core.agent.LLMChain") as mock:
        mock_instance = mock.return_value
        mock_instance.arun = AsyncMock(return_value="This is a test response")
        yield mock


@pytest.fixture
def mock_memory_store():
    """Create a fake memory store so tests do not talk to Redis."""
    with patch("app.core.agent.RedisMemoryStore") as mock:
        mock_instance = mock.return_value
        mock_instance.get_history.return_value = MagicMock(name="conversation_memory")
        yield mock_instance


@pytest.fixture
def agent(mock_llm_service, mock_qdrant_service, mock_memory_store, mock_llm_chain):
    """Create a real AIAgent whose external collaborators are all mocked."""
    with patch("app.core.agent.QdrantService", return_value=mock_qdrant_service), patch(
        "app.core.agent.RedisMemoryStore", return_value=mock_memory_store
    ):
        yield AIAgent(mock_llm_service)


class TestAIAgent:
    """Tests for the orchestration logic inside AIAgent."""

    @pytest.mark.asyncio
    async def test_process_message(self, agent):
        """process_message should search, run the chain, and return mapped sources."""
        response, sources = await agent.process_message(
            "What is artificial intelligence?", "test_session"
        )

        assert response == "This is a test response"
        assert len(sources) == 1
        assert sources[0]["title"] == "Knowledge Base Entry"
        agent.memory_store.get_history.assert_called_once_with("test_session")
        agent.qdrant_service.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_builds_chain_with_memory_and_context(
        self, agent, mock_llm_chain
    ):
        """The chain should be built with prompt + memory and run with input + RAG context."""
        await agent.process_message("Test prompt", "test_session")

        mock_llm_chain.assert_called_once_with(
            llm=agent.llm,
            prompt=agent.prompt_template,
            memory=agent.memory_store.get_history.return_value,
        )
        mock_llm_chain.return_value.arun.assert_called_once()
        _, run_kwargs = mock_llm_chain.return_value.arun.call_args
        assert run_kwargs["input"] == "Test prompt"
        assert "Knowledge Base Entry" in run_kwargs["context"]

    @pytest.mark.asyncio
    async def test_process_message_deduplicates_sources(self, agent):
        """Duplicate retrieval hits should collapse into a single source entry."""
        agent.qdrant_service.search.return_value = [
            {
                "title": "test.txt",
                "content": "Alyssa built a prototype AI agent backend.",
                "url": "kb://alyssa",
                "score": 0.92,
            },
            {
                "title": "test.txt",
                "content": "Alyssa built a prototype AI agent backend.",
                "url": "kb://alyssa",
                "score": 0.91,
            },
        ]

        _, sources = await agent.process_message("What did Alyssa build?", "test_session")

        assert len(sources) == 1
        assert sources[0]["title"] == "test.txt"
        assert sources[0]["url"] == "kb://alyssa"

    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Exceptions from the chain should bubble up to the caller."""
        with patch("app.core.agent.LLMChain") as mock_chain:
            mock_chain.return_value.arun = AsyncMock(side_effect=Exception("Test error"))

            with pytest.raises(Exception, match="Test error"):
                await agent.process_message("Test error message", "test_session")

    @pytest.mark.asyncio
    async def test_memory_interaction(self, agent):
        """The agent should rely on chain-managed memory instead of manual writes."""
        await agent.process_message("Test memory", "test_session")

        agent.memory_store.get_history.assert_called_once_with("test_session")
        agent.memory_store.add_interaction.assert_not_called()

    def test_reset(self, agent):
        """reset should delegate directly to the memory store."""
        agent.reset("test_session")

        agent.memory_store.clear_history.assert_called_once_with("test_session")
