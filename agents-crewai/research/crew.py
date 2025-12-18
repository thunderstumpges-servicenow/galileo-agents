"""Research Crew - Researcher and Analyst agents with knowledge base search."""
import os
import sys
import truststore

from crewai import Crew, LLM, Task
from crewai.tools import tool
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from opentelemetry import trace
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from traceloop.sdk import Traceloop

from agents import create_analyst_agent, create_researcher_agent
from shared import logger
from tools import create_knowledge_retriever, search_knowledge_base

truststore.inject_into_ssl()
load_dotenv()

Traceloop.init(
    app_name="research-crew",
    resource_attributes={
        "galileo.project.name": "galileo-agents",
        "galileo.logstream.name": "research-crew",
    },
    disable_batch=True,
)

# Add console exporter for debugging if enabled
if os.getenv("TRACELOOP_CONSOLE_EXPORTER_ENABLED", "false").lower() == "true":
    tracer_provider = trace.get_tracer_provider()
    if hasattr(tracer_provider, "add_span_processor"):
        console_processor = BatchSpanProcessor(ConsoleSpanExporter())
        tracer_provider.add_span_processor(console_processor)

# Knowledge base setup
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
retriever = create_knowledge_retriever(embeddings)


@tool
def knowledge_search(query: str) -> str:
    """Search the knowledge base for relevant information."""
    return search_knowledge_base(query, retriever)


def create_crew(topic: str):
    llm = LLM(model="gpt-4o-mini", temperature=0.3)
    researcher = create_researcher_agent(llm, tools=[knowledge_search])
    analyst = create_analyst_agent(llm)

    research_task = Task(
        description=f"Research the topic: {topic}. Use knowledge_search to find relevant information.",
        expected_output="A comprehensive collection of research findings with key facts.",
        agent=researcher,
    )
    analysis_task = Task(
        description="Analyze the research findings and create a summary with key insights.",
        expected_output="A well-structured analysis report with findings and recommendations.",
        agent=analyst,
        context=[research_task],
    )

    return Crew(agents=[researcher, analyst], tasks=[research_task, analysis_task], verbose=True)


def main(topic: str = "Benefits of microservices architecture"):
    logger.info(f"Research Crew - Topic: {topic}")
    crew = create_crew(topic)
    result = crew.kickoff()
    logger.info(f"Result: {result}")
    return result


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Benefits of microservices architecture"
    main(topic)