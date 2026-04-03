from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Tuple
import logging

from app.core.memory import RedisMemoryStore
from app.services.qdrant_service import QdrantService
from app.utils.embeddings import get_embedding
from app.utils.ranking import rank_results, filter_results

logger = logging.getLogger(__name__)

class AIAgent:
    """Main AI agent that orchestrates the conversation flow and information retrieval"""

    def __init__(self, llm_service):
        """Initialize the AI agent with necessary services

        Args:
            llm_service: Service providing access to the language model
        """
        self.llm = llm_service.get_llm()
        self.memory_store = RedisMemoryStore()
        self.qdrant_service = QdrantService()

        # Define the prompt template for the LLM
        self.prompt_template = PromptTemplate.from_template(
            '''You are a helpful AI assistant for analytics-related questions.

            Context from previous conversations:
            {history}

            Relevant information from the knowledge base:
            {context}

            Human: {input}
            AI: '''
        )

    async def process_message(self, message: str, session_id: str) -> Tuple[str, List[Dict[str, str]]]:
        """Process a user message and generate a response

        Args:
            message: User's message
            session_id: Session ID for tracking conversation history

        Returns:
            Tuple containing the response text and list of sources
        """
        try:
            # Retrieve conversation history
            history = self.memory_store.get_history(session_id)

            # Generate embedding for the message
            embedding = get_embedding(message)

            # Retrieve relevant context from vector store (RAG)
            rag_results = self.qdrant_service.search(embedding)

            # Rank and filter retrieved documents before passing them to the LLM
            ranked_results = rank_results(rag_results, embedding)
            filtered_results = filter_results(ranked_results)

            # Build context string from top results
            context = "\n\n".join([
                f"Source: {r.get('title', 'Unknown')}\n{r.get('content', '')}"
                for r in filtered_results[:5]
            ])

            # Create the chain
            chain = ConversationChain(
                llm=self.llm,
                prompt=self.prompt_template,
                memory=history
            )

            # Run the chain to generate response
            response = await chain.arun(input=message, context=context)

            # Extract source information
            sources = [
                {
                    "title": r.get("title", "Unknown"),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")[:150] + "..." if len(r.get("content", "")) > 150 else r.get("content", "")
                }
                for r in filtered_results[:5]
            ]

            # Store the interaction in memory
            self.memory_store.add_interaction(session_id, message, response)

            return response, sources

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise

    def reset(self, session_id: str) -> None:
        """Reset the agent's state for a session

        Args:
            session_id: Session ID to reset
        """
        self.memory_store.clear_history(session_id)
