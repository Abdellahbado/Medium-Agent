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

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

search_tool = TavilySearchResults()


class Section(BaseModel):
    name: str
    description: str
    search_query: Optional[List[str]] = None  


class Sections(BaseModel):
    sections: List[Section]


planner = llm.with_structured_output(Sections)


class State(TypedDict):
    topic: str
    sections: List[Section]
    language_tone: str            # e.g., "formal", "simple", "professional"
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
                    "3. An optional list of search queries if additional research is necessary (the search queries must include any URLs provided in the topic).\n\n"
                    "Ensure that your plan encourages the use of external search results wherever possible to verify and update the information."
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
                "and include them as references, enhancing the content. If search results are provided, use "
                "them to correct inaccuracies and add up-to-date information to your narrative."
            )
        ),
        HumanMessage(
            content=f"Section: {section.name}\nDescription: {section.description}"
        ),
    ]

    if section.search_query:
        for query in section.search_query:
            search_results = search_tool.run(query)
            if search_results:
                messages.append(
                    HumanMessage(
                        content=(
                            f"Search Results for query '{query}':\n{search_results}\n\n"
                            "Use these results to verify facts, correct any inaccuracies, and enrich your content."
                        )
                    )
                )
            else:
                messages.append(
                    HumanMessage(
                        content=f"No search results were found for the query: '{query}'."
                    )
                )

    content = llm.invoke(messages)
    return {"completed_sections": [content.content]}


def final_report_generator(state: State):
    
    tone = state.get("language_tone", "professional")
    combined_content = "\n\n---\n\n".join(state["completed_sections"])
    messages = [
        SystemMessage(
            content=(
                f"You are an expert writer. Generate the final report in a {tone} tone. "
                "Ensure that the language style throughout the report reflects this tone consistently."
            )
        ),
        HumanMessage(
            content=f"Combine the following content into a coherent final report:\n{combined_content}"
        ),
    ]
    final_report_response = llm.invoke(messages)
    return {"final_report": final_report_response.content}


def assign_workers(state: State):
    return [Send("llm_writer", {"section": s}) for s in state["sections"]]


builder = StateGraph(State)

builder.add_node("orchestrator", orchestrator)
builder.add_node("llm_writer", llm_writer)
builder.add_node("final_report_generator", final_report_generator)

builder.add_edge(START, "orchestrator")
builder.add_conditional_edges("orchestrator", assign_workers)
builder.add_edge("llm_writer", "final_report_generator")
builder.add_edge("final_report_generator", END)

workflow = builder.compile()
