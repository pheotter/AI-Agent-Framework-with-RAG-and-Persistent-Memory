import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from app.services.qdrant_service import QdrantService
from app.utils.embeddings import batch_get_embeddings, get_embedding
from app.utils.ranking import rank_results


@pytest.fixture
def mock_qdrant_client():
    """Create a fake Qdrant client so tests never hit a real vector DB."""
    with patch("app.services.qdrant_service.QdrantClient") as mock:
        mock_instance = mock.return_value

        # Simulate an empty Qdrant server so collection creation is exercised.
        collections_response = MagicMock()
        collections_response.collections = []
        mock_instance.get_collections.return_value = collections_response

        # Simulate two hits returned by a vector search.
        mock_instance.search.return_value = [
            MagicMock(
                id="doc1",
                payload={"content": "Test content 1", "title": "Test document 1"},
                score=0.9,
            ),
            MagicMock(
                id="doc2",
                payload={"content": "Test content 2", "title": "Test document 2"},
                score=0.7,
            ),
        ]

        yield mock_instance


@pytest.fixture
def qdrant_service(mock_qdrant_client):
    """Create a real QdrantService backed by the fake client."""
    with patch("app.services.qdrant_service.QdrantClient", return_value=mock_qdrant_client):
        return QdrantService()


@pytest.fixture
def mock_embedding_model():
    """Create a fake embedding model for single and batch embedding tests."""
    with patch("app.utils.embeddings.embeddings_model") as mock:
        mock.embed_query.return_value = [0.1] * 1536
        mock.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]
        yield mock


class TestRAGSystem:
    """Tests for the retrieval and ranking helpers used by the RAG flow."""

    def test_embedding_generation(self, mock_embedding_model):
        """get_embedding should call the single-text embedding API."""
        embedding = get_embedding("Test query")

        mock_embedding_model.embed_query.assert_called_once_with("Test query")
        assert len(embedding) == 1536

    def test_batch_embedding_generation(self, mock_embedding_model):
        """batch_get_embeddings should call the batch embedding API."""
        texts = ["Text 1", "Text 2"]
        embeddings = batch_get_embeddings(texts)

        mock_embedding_model.embed_documents.assert_called_once_with(texts)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536

    def test_qdrant_search(self, qdrant_service):
        """search should map raw Qdrant hits into the app's result schema."""
        results = qdrant_service.search([0.1] * 1536)

        qdrant_service.client.search.assert_called_once()
        assert len(results) == 2
        assert results[0]["title"] == "Test document 1"
        assert results[0]["content"] == "Test content 1"
        assert results[0]["score"] == 0.9

    def test_result_ranking(self):
        """rank_results should combine semantic similarity with the source score."""
        results = [
            {"content": "Test content 1", "title": "Test document 1", "score": 0.7, "embedding": [0.2] * 1536},
            {"content": "Test content 2", "title": "Test document 2", "score": 0.9, "embedding": [0.3] * 1536},
            {"content": "Test content 3", "title": "Test document 3", "score": 0.5, "embedding": [0.1] * 1536},
        ]
        query_embedding = [0.1] * 1536

        ranked = rank_results(results, query_embedding)

        # All three vectors point in the same direction, so their cosine
        # similarity is effectively equal. In that case the higher Qdrant
        # score should dominate the final ranking.
        assert ranked[0]["title"] == "Test document 2"
        assert ranked[1]["title"] == "Test document 1"
        assert ranked[2]["title"] == "Test document 3"

    def test_qdrant_collection_creation(self, qdrant_service):
        """Initialization should create the collection when it is missing."""
        qdrant_service.client.create_collection.assert_called_once()

    def test_qdrant_add_document(self, qdrant_service):
        """add_document should send one point to Qdrant via upsert."""
        qdrant_service.add_document(
            str(uuid4()),
            [0.1] * 1536,
            {"content": "Test content", "title": "Test document"},
        )

        qdrant_service.client.upsert.assert_called_once()

    def test_qdrant_add_documents(self, qdrant_service):
        """add_documents should batch points and send them via upsert."""
        documents = [
            {"id": str(uuid4()), "embedding": [0.1] * 1536, "metadata": {"content": "Content 1"}},
            {"id": str(uuid4()), "embedding": [0.2] * 1536, "metadata": {"content": "Content 2"}},
        ]

        qdrant_service.add_documents(documents)

        qdrant_service.client.upsert.assert_called_once()
