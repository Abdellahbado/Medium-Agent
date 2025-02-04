from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from langgraph.constants import Send, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Annotated, List, Optional, TypedDict
from pydantic import BaseModel
import operator
import os
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults

load_dotenv()

# Initialize ChatGroq LLM with increased creativity
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# Initialize search tool
search_tool = TavilySearchResults()


class Section(BaseModel):
    name: str
    description: str
    search_query: Optional[str] = None  # Optional search query for the section


class Sections(BaseModel):
    sections: List[Section]


# Augment LLM with structured output for planning
planner = llm.with_structured_output(Sections)


# Graph state definition
class State(TypedDict):
    topic: str
    sections: List[Section]
    completed_sections: Annotated[List[str], operator.add]
    final_report: str


class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[List[str], operator.add]


# Nodes
def orchestrator(state: State):
    report_plan = planner.invoke(
        [
            SystemMessage(
                content=(
                    "You are a meticulous content planner for in-depth articles. "
                    "Generate a report plan with fewer than 3 sections. Each section must include:\n"
                    "1. A clear and concise name (required).\n"
                    "2. A detailed description outlining what the section should cover (required).\n"
                    "3. An optional search query if additional research is necessary.\n\n"
                    "Ensure that your plan encourages the use of external search results wherever possible "
                    "to verify and update the information."
                )
            ),
            HumanMessage(content=f"Topic: {state['topic']}"),
        ]
    )
    return {"sections": report_plan.sections}


def llm_writer(state: WorkerState):
    section = state["section"]
    messages = [
        SystemMessage(
            content=(
                "You are an expert writer tasked with creating a detailed and accurate report section. "
                "Make sure to incorporate external search results as a primary source for verifying facts "
                "and include them as references"
                "and enhancing the content. If search results are provided, use them to correct inaccuracies "
                "and add up-to-date information to your narrative."
            )
        ),
        HumanMessage(
            content=f"Section: {section.name}\nDescription: {section.description}"
        ),
    ]

    if section.search_query:
        search_results = search_tool.run(section.search_query)
        if search_results:
            messages.append(
                HumanMessage(
                    content=(
                        f"Search Results for '{section.search_query}':\n{search_results}\n\n"
                        "Use these results to verify facts, correct any inaccuracies, and enrich your content."
                    )
                )
            )
        else:
            messages.append(
                HumanMessage(
                    content=(
                        "No search results were found for the specified query. "
                        "If possible, try to consider related topics or rephrase the query in your writing."
                    )
                )
            )

    content = llm.invoke(messages)
    return {"completed_sections": [content.content]}


def synthesizer(state: State):
    """Combine all sections into the final report"""
    return {"final_report": "\n\n---\n\n".join(state["completed_sections"])}


# Conditional edge function to create worker nodes
def assign_workers(state: State):
    """Create a worker node for each section"""
    return [Send("llm_writer", {"section": s}) for s in state["sections"]]


# Build workflow
builder = StateGraph(State)

# Add nodes
builder.add_node("orchestrator", orchestrator)
builder.add_node("llm_writer", llm_writer)
builder.add_node("synthesizer", synthesizer)

# Define edges
builder.add_edge(START, "orchestrator")
builder.add_conditional_edges("orchestrator", assign_workers)
builder.add_edge("llm_writer", "synthesizer")
builder.add_edge("synthesizer", END)

# Compile workflow
workflow = builder.compile()

