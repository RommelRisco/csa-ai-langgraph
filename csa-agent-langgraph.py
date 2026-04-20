from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

SYSTEM_MESSAGE = SystemMessage(
    content=(
        "You are an AI customer support agent. "
        "Help customers professionally and accurately within your area of expertise. "
        "If a topic or request falls outside the scope of customer support "
        "(Technical, Billing, or General inquiries), escalate it to a human agent."
    )
)


class State(TypedDict):
    query: str
    category: str
    sentiment: str
    request_type: str
    response: str

def categorize(state: State) -> State:
    prompt = ChatPromptTemplate.from_messages([
        SYSTEM_MESSAGE,
        ("human", "Categorize the following customer query into one of these categories: "
                  "Technical, Billing, General. "
                  "Respond with only the category name and nothing else. Query: {query}"),
    ])
    chain = prompt | ChatOpenAI(temperature=0)
    category = chain.invoke({"query": state["query"]}).content
    return {"category": category}


def analyze_sentiment(state: State) -> State:
    prompt = ChatPromptTemplate.from_messages([
        SYSTEM_MESSAGE,
        ("human", "Analyze the sentiment of the following customer query. "
                  "Respond with only the word Positive, Neutral, or Negative and nothing else. Query: {query}"),
    ])
    chain = prompt | ChatOpenAI(temperature=0)
    sentiment = chain.invoke({"query": state["query"]}).content
    return {"sentiment": sentiment}


def handle_technical(state: State) -> State:
    prompt = ChatPromptTemplate.from_messages([
        SYSTEM_MESSAGE,
        ("human", "Provide a technical support response to the following query: {query}"),
    ])
    chain = prompt | ChatOpenAI(temperature=0)
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}


def handle_billing(state: State) -> State:
    prompt = ChatPromptTemplate.from_messages([
        SYSTEM_MESSAGE,
        ("human", "Provide a billing support response to the following query: {query}"),
    ])
    chain = prompt | ChatOpenAI(temperature=0)
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}


def handle_general(state: State) -> State:
    prompt = ChatPromptTemplate.from_messages([
        SYSTEM_MESSAGE,
        ("human", "Provide a general support response to the following query: {query}"),
    ])
    chain = prompt | ChatOpenAI(temperature=0)
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}


def classify_request_type(state: State) -> State:
    prompt = ChatPromptTemplate.from_messages([
        SYSTEM_MESSAGE,
        ("human", "Classify the following customer query into one of these ITSM request types: "
                  "Incident, Service Request, Change, Problem. "
                  "Incident: something is broken or not working. "
                  "Service Request: asking for something new or a standard service. "
                  "Change: requesting a modification to an existing system or configuration. "
                  "Problem: recurring or root-cause investigation of an issue. "
                  "Respond with only the type name. Query: {query}"),
    ])
    chain = prompt | ChatOpenAI(temperature=0)
    request_type = chain.invoke({"query": state["query"]}).content
    return {"request_type": request_type}


def escalate(state: State) -> State:
    # TODO- Define escalation logic through 
    return {"response": "Your query has been escalated to a human agent."}


def router(state: State) -> str:
    sentiment = state["sentiment"].strip()
    category = state["category"].strip()

    if "negative" in sentiment.lower():
        return "escalate"

    if "technical" in category.lower():
        return "handle_technical"
    elif "billing" in category.lower():
        return "handle_billing"
    elif "general" in category.lower():
        return "handle_general"

    return "escalate"


workflow = StateGraph(State)

workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("classify_request_type", classify_request_type)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalate", escalate)

workflow.set_entry_point("categorize")

workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_edge("analyze_sentiment", "classify_request_type")
workflow.add_conditional_edges(
    "classify_request_type",
    router,
    {
        "handle_technical": "handle_technical",
        "handle_billing": "handle_billing",
        "handle_general": "handle_general",
        "escalate": "escalate",
    },
)

workflow.add_edge("handle_technical", END)
workflow.add_edge("handle_billing", END)
workflow.add_edge("handle_general", END)
workflow.add_edge("escalate", END)

app = workflow.compile()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python csa-langraph.py \"<your query>\"")
        sys.exit(1)
    query = sys.argv[1]
    result = app.invoke({"query": query})
    print("Category:", result["category"])
    print("Sentiment:", result["sentiment"])
    print("Request Type:", result["request_type"])
    print("Response:", result["response"])
