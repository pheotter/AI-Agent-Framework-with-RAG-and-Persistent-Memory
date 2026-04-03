# Architecture Overview

## Purpose

This project demonstrates a minimal AI agent backend that combines:

- an API layer for user interaction
- a retrieval layer for grounding responses on domain documents
- a memory layer for session continuity across turns

The overall design is intentionally simple so the request flow is easy to trace in code.

## Main Components

### API Layer

- `app/main.py` creates the FastAPI application.
- `app/routes/chat.py` exposes the chat endpoints.

Responsibilities:

- validate incoming requests
- create an agent instance
- return structured response payloads

### Agent Orchestration

- `app/core/agent.py` is the main control-flow module.

Responsibilities:

- load session memory
- embed the incoming query
- retrieve relevant knowledge from Qdrant
- rank and filter retrieved results
- construct prompt inputs
- call the LLM
- persist the interaction

### Memory Layer

- `app/core/memory.py` wraps Redis-backed chat history.

Responsibilities:

- maintain per-session conversation state
- support multi-turn interactions through shared `session_id`
- clear history when requested

### Retrieval Layer

- `app/services/qdrant_service.py` manages vector store operations.
- `app/utils/embeddings.py` generates embeddings for queries and documents.
- `app/utils/ranking.py` reranks and filters retrieved results.

Responsibilities:

- initialize the vector collection
- add embedded document chunks
- search for semantically similar chunks
- return metadata used in final responses

### Ingestion Layer

- `scripts/seed_knowledge.py` loads local files, chunks them, embeds them, and inserts them into Qdrant.

Responsibilities:

- read source documents
- split long files into smaller chunks
- create embeddings in batch
- write the resulting points into the vector store

## Request Flow

```text
Client
  -> POST /chat/
  -> app/routes/chat.py
  -> app/core/agent.py
     -> app/core/memory.py
     -> app/utils/embeddings.py
     -> app/services/qdrant_service.py
     -> app/utils/ranking.py
     -> app/services/llm_service.py
  <- response + sources + session_id
```

## Data Flow

### Chat request path

1. The user sends `message` and optionally `session_id`.
2. The route creates `AIAgent`.
3. The agent loads history from Redis.
4. The agent embeds the user message.
5. The embedding is used to search Qdrant.
6. Retrieved chunks are ranked and filtered.
7. Prompt inputs are assembled from:
   - chat history
   - retrieved context
   - current user input
8. The LLM generates a response.
9. The interaction remains associated with the same Redis session.

### Knowledge-base ingestion path

1. Local documents are read from `data/knowledge_base/`.
2. Each file is split into chunks.
3. Each chunk is embedded.
4. Chunks and metadata are inserted into Qdrant.
5. Future user queries retrieve from that stored collection.

## Why These Choices

- FastAPI keeps the API surface small and readable.
- LangChain reduces prompt and conversation wiring for a prototype.
- Qdrant provides a dedicated vector store instead of in-memory retrieval.
- Redis gives simple session persistence for multi-turn chat.

## Prototype Boundaries

This architecture is useful for demos, learning, and portfolio discussion, but it does not yet cover:

- auth and user isolation beyond session IDs
- observability and tracing
- prompt/version management
- retrieval evaluation
- production deployment concerns
