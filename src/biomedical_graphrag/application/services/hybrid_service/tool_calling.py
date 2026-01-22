import json
from typing import Any

from openai import OpenAI

from biomedical_graphrag.application.services.hybrid_service.neo4j_query import Neo4jGraphQuery
from biomedical_graphrag.application.services.hybrid_service.prompts.hybrid_prompts import (
    HYBRID_PROMPT,
    fusion_summary_prompt,
)
from biomedical_graphrag.application.services.hybrid_service.tools.enrichment_tools import (
    ENRICHMENT_TOOLS,
)
from biomedical_graphrag.config import settings
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


openai_client = OpenAI(api_key=settings.openai.api_key.get_secret_value())


def get_neo4j_schema() -> str:
    """Retrieve the Neo4j schema dynamically."""
    neo4j = Neo4jGraphQuery()
    schema = neo4j.get_schema()
    logger.info(f"Retrieved Neo4j schema length: {len(schema)}")
    return schema


# --------------------------------------------------------------------
# Phase 1 — Tool selection + execution
# --------------------------------------------------------------------
def run_graph_enrichment(question: str, qdrant_chunks: list[str]) -> dict[str, Any]:
    """Run graph enrichment.

    Args:
        question: The user question.
        qdrant_chunks: The Qdrant chunks.

    Returns:
        The Neo4j results.
    """
    schema = get_neo4j_schema()
    neo4j = Neo4jGraphQuery()

    prompt = HYBRID_PROMPT.format(schema=schema, question=question, chunks="---".join(qdrant_chunks))
    logger.info(f"Prompt for tool selection based on question and Qdrant retrieval:\n{prompt}")

    response = openai_client.responses.create(  # type: ignore[call-overload]
        model=settings.openai.model,
        tools=ENRICHMENT_TOOLS,
        input=[{"role": "user", "content": prompt}],
        tool_choice="auto",
    )

    results = {}
    if response.output:
        for tool_call in response.output:
            if tool_call.type == "function_call":
                name = tool_call.name
                args = (
                    json.loads(tool_call.arguments)
                    if isinstance(tool_call.arguments, str)
                    else tool_call.arguments
                )
                func = getattr(neo4j, name, None)
                if func:
                    try:
                        logger.info(f"LLM tool_call: {tool_call.name} args={tool_call.arguments}")
                        results[name] = func(**args)
                    except Exception as e:
                        results[name] = f"Error: {e}"

    logger.info(f"Neo4j tool results: {results}")
    return results


# --------------------------------------------------------------------
# Phase 2 — Fusion summarization
# --------------------------------------------------------------------
def summarize_fused_results(question: str, qdrant_chunks: list[str], neo4j_results: dict) -> str:
    """Fuse semantic and graph evidence into one final biomedical summary.

    Args:
        question: The user question.
        qdrant_chunks: The Qdrant chunks.
        neo4j_results: The Neo4j results.

    Returns:
        The summarized results.
    """
    prompt = fusion_summary_prompt(question, qdrant_chunks, neo4j_results)
    resp = openai_client.responses.create(
        model=settings.openai.model,
        input=prompt,
        temperature=settings.openai.temperature,
        max_output_tokens=settings.openai.max_tokens,
    )
    return resp.output_text.strip()


# --------------------------------------------------------------------
# Unified helper
# --------------------------------------------------------------------
def run_graph_enrichment_and_summarize(question: str, qdrant_chunks: list[str]) -> str:
    """Run graph enrichment and summarize the results.

    Args:
        question: The user question.
        qdrant_chunks: The Qdrant chunks.

    Returns:
        The summarized results.
    """
    neo4j_results = run_graph_enrichment(question, qdrant_chunks)
    return summarize_fused_results(question, qdrant_chunks, neo4j_results)
