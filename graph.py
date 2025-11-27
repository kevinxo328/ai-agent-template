from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from pydantic_ai import Agent
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessagesTypeAdapter,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
)

from agents.chat_agent import create_chat_agent
from utils.message import StreamMessage, UserMessage, to_model_message
from utils.model import OpenAIModelConfig


class State(TypedDict):
    user_input: str
    model_config: OpenAIModelConfig

    # Store all messages, including tool calls from Pydantic AI, as a list of bytes for easy database storage.
    messages: Annotated[list[bytes], lambda a, b: a + b]


async def chat(state: State):
    agent = create_chat_agent(state["model_config"])
    messages = [
        ModelMessagesTypeAdapter.dump_json(
            [to_model_message(UserMessage(content=state["user_input"]))]
        )
    ]

    # Create a stream writer to capture the output for langgraph
    writer = get_stream_writer()
    final_message = None
    async with agent.iter(
        user_prompt=state["user_input"],
        message_history=[
            msg
            for m in state["messages"]
            for msg in ModelMessagesTypeAdapter.validate_json(m)
        ],
    ) as run:
        async for node in run:
            if Agent.is_user_prompt_node(node) or Agent.is_end_node(node):
                continue

            elif Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as request_stream:
                    final_result_found = False
                    async for event in request_stream:
                        if isinstance(event, PartStartEvent):
                            print(
                                f"[Request] Starting part {event.index}: {event.part!r}"
                            )
                        elif isinstance(event, PartDeltaEvent):
                            if isinstance(event.delta, TextPartDelta):
                                print(
                                    f"[Request] Part {event.index} text delta: {event.delta.content_delta!r}"
                                )
                            elif isinstance(event.delta, ThinkingPartDelta):
                                # print(
                                #     f"[Request] Part {event.index} thinking delta: {event.delta.content_delta!r}"
                                # )
                                writer(
                                    StreamMessage(
                                        type="thinking",
                                        content_delta=event.delta.content_delta,
                                        timestamp=request_stream.timestamp().isoformat(),
                                    )
                                )
                            elif isinstance(event.delta, ToolCallPartDelta):
                                print(
                                    f"[Request] Part {event.index} args delta: {event.delta.args_delta}"
                                )
                        elif isinstance(event, FinalResultEvent):
                            final_result_found = True
                            break

                    if final_result_found:
                        # Once the final result is found, we can call `AgentStream.stream_text()` to stream the text.
                        # A similar `AgentStream.stream_output()` method is available to stream structured output.
                        async for output in request_stream.stream_text():
                            # print(f"[Final Result] {output}")
                            final_message = StreamMessage(
                                type="text",
                                content=output,
                                timestamp=request_stream.timestamp().isoformat(),
                            )
                            writer(final_message)

            elif Agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as handle_stream:
                    async for event in handle_stream:
                        if isinstance(event, FunctionToolCallEvent):
                            # print(
                            #     f"[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})"
                            # )
                            writer(
                                StreamMessage(
                                    type="tool_call_start",
                                    tool_call_name=event.part.tool_name,
                                    tool_call_args=event.part.args,
                                    tool_call_id=event.part.tool_call_id,
                                    timestamp=node.model_response.timestamp.isoformat(),
                                )
                            )
                        elif isinstance(event, FunctionToolResultEvent):
                            # print(
                            #     f"[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}"
                            # )
                            writer(
                                StreamMessage(
                                    type="tool_call_end",
                                    tool_call_id=event.tool_call_id,
                                    content=str(event.result.content),
                                    timestamp=node.model_response.timestamp.isoformat(),
                                )
                            )

    if final_message is not None:
        messages.append(
            ModelMessagesTypeAdapter.dump_json([to_model_message(final_message)])
        )

    return {"messages": messages}


graph_builder = StateGraph(State)

graph_builder.add_node("chat", chat)
graph_builder.add_edge(START, "chat")
graph_builder.add_edge("chat", END)

# checkpointer = InMemorySaver()
graph = graph_builder.compile(name="Chat Graph")
