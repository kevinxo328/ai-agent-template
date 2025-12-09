import operator
from datetime import datetime
from typing import Literal

from langchain.agents import create_agent
from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from agents.demo.env import env
from models.openai.langchain import create_openai_model
from models.schema import OpenAIModelConfig

config = OpenAIModelConfig(
    model_name=env.OPENAI_MODEL_NAME,
    api_key=env.OPENAI_API_KEY,
    base_url=env.OPENAI_BASE_URL,
)
model = create_openai_model(config)
search_tool = TavilySearch(max_results=5)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    intent: str
    search_query: str


class IntentClassification(BaseModel):
    intent: Literal["web_search", "chat"] = Field(
        ...,
        description="The classified intent of the user message, web_search=requires web search, chat=general chat",
    )


def intent_classification_node(state: MessagesState):
    """Classify the user's intent"""
    # Use a structured output model to classify intent, to make sure the output is constrained.
    structured_model = model.with_structured_output(IntentClassification)
    system_prompt = """
    You are an intent classification model. Classify the user's intent based on the message.

    If the user is asking a question that requires latest information, return "web_search".
    Else return "chat".
    """

    for msg in state["messages"]:
        print(msg)

    # Only use the last 5 messages for intent classification
    user_messages = [
        msg
        for msg in state["messages"]
        if isinstance(msg, AIMessage)
        or isinstance(msg, HumanMessage)
        or (isinstance(msg, dict) and msg["type"] == "human")
        or (isinstance(msg, dict) and msg["type"] == "ai")
    ][-5:]

    decision = structured_model.invoke(
        [
            SystemMessage(content=system_prompt),
        ]
        + user_messages
    )

    return {
        "intent": decision.intent,
    }


def chat_node(state: MessagesState):
    """Handle general chat messages"""

    # Trim messages to fit within model context window
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        max_tokens=5000,
        token_counter=count_tokens_approximately,
        start_on="human",
        end_on=("human", "tool"),
    )

    return {
        "messages": [model.invoke(trimmed_messages)],
    }


def rewrite_query_node(state: MessagesState):
    """Rewrite the user query for web search"""
    system_prompt = """
    You are a query rewriting model. Rewrite the user's query to be more suitable for web search.
    Focus on keywords and important phrases.

    The current date is {datetime_now}. 

    <Rules>
    - Don't use quotation marks.
    - If the date is mentioned in the user query, make sure to include it in the rewritten query. And use the format YYYY-MM-DD.
    <Rules>
    """

    user_messages = [
        msg
        for msg in state["messages"]
        if isinstance(msg, HumanMessage)
        or (isinstance(msg, dict) and msg["type"] == "human")
    ][-5:]
    rewrite_query = model.invoke(
        [
            SystemMessage(
                content=system_prompt.format(datetime_now=datetime.now().date())
            ),
        ]
        + user_messages
    )

    return {
        "search_query": rewrite_query.content,
    }


def web_search_node(state: MessagesState):
    """Handle web search messages"""
    results = search_tool.run(state["seach_query"])
    search_results = "\n\n".join(
        [f"[{r['title']}] {r['content']}" for r in results["results"]]
    )
    content = (
        f"Web Search Results for query '{state['seach_query']}':\n\n{search_results}"
    )
    return {"messages": [AIMessage(content=content)]}


def web_search_agent_node(state: MessagesState):
    """Handle web search messages"""
    agent = create_agent(
        model=model,
        tools=[search_tool],
        system_prompt="You are a helpful assistant that uses web search to answer user queries.",
    )

    response = agent.invoke({"messages": [HumanMessage(content=state["search_query"])]})
    print(response)

    return {"messages": response["messages"][-1:]}


# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("intent_classification", intent_classification_node)
agent_builder.add_node("chat_node", chat_node)
agent_builder.add_node("rewrite_query_node", rewrite_query_node)
agent_builder.add_node("web_search_agent_node", web_search_agent_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "intent_classification")
agent_builder.add_conditional_edges(
    "intent_classification",
    lambda state: state["intent"],
    {
        "web_search": "rewrite_query_node",
        "chat": "chat_node",
    },
)
agent_builder.add_edge("rewrite_query_node", "web_search_agent_node")
agent_builder.add_edge("chat_node", END)
agent_builder.add_edge("web_search_agent_node", END)

# Compile the agent
agent = agent_builder.compile()
