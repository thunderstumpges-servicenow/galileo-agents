"""Calculator Agent - Performs calculations and unit conversions."""
import os
import sys
import truststore

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from opentelemetry import trace
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from traceloop.sdk import Traceloop

from prompt import CALCULATOR_AGENT_SYSTEM_PROMPT
from shared import logger
from tools import calculate, convert_units

truststore.inject_into_ssl()

os.environ["OTEL_EXPORTER_OTLP_INSECURE"] = "true"

load_dotenv()

os.environ["OTEL_EXPORTER_OTLP_INSECURE"] = "true"

Traceloop.init(
    app_name="calculator-agent",
    resource_attributes={
        "galileo.project.name": "galileo-agents",
        "galileo.logstream.name": "calculator-agent",
    },
    disable_batch=True,
)

# Add console exporter for debugging if enabled
if os.getenv("TRACELOOP_CONSOLE_EXPORTER_ENABLED", "false").lower() == "true":
    tracer_provider = trace.get_tracer_provider()
    if hasattr(tracer_provider, "add_span_processor"):
        console_processor = BatchSpanProcessor(ConsoleSpanExporter())
        tracer_provider.add_span_processor(console_processor)


@tool
def calc_tool(expression: str) -> str:
    """Evaluate a mathematical expression."""
    result = calculate(expression)
    if result["success"]:
        return f"{result['expression']} = {result['result']}"
    return f"Error: {result['error']}"


@tool
def convert_tool(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between units."""
    result = convert_units(value, from_unit, to_unit)
    if result["success"]:
        return f"{result['original']} = {result['converted']}"
    return f"Error: {result['error']}"


def create_calculator_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return create_agent(llm, [calc_tool, convert_tool], system_prompt=CALCULATOR_AGENT_SYSTEM_PROMPT)


def main(query: str = "Convert 100 km to mi"):
    logger.info(f"Calculator Agent - Query: {query}")
    agent = create_calculator_agent()

    current_span = trace.get_current_span()
    # Add attributes to the current span
    current_span.set_attribute("gen_ai.agent.id", "thunder-calculator-agent-langgraph")
    current_span.set_attribute("discovery.ci", "thunder-calculator-agent-langgraph")

    result = agent.invoke({"messages": [("user", query)]})
    response = result["messages"][-1].content
    logger.info(f"Response: {response}")
    return response


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Convert 100 km to mi"
    main(query)