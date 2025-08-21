import asyncio

import streamlit as st
from langchain_core.runnables.config import RunnableConfig
from langfuse.langchain import CallbackHandler
from streamlit_chatbox import ChatBox, Markdown

from graph import OpenAIModelConfig, State, checkpointer, graph
from utils.langfuse import langfuse
from utils.message import StreamMessage

THREAD_ID = "1"


async def render_streamlit():
    # Sidebar for OpenAI-compatible API settings
    with st.sidebar:
        st.sidebar.header("OpenAI API Settings")
        api_key = st.sidebar.text_input("API Key", type="password")
        base_url = st.sidebar.text_input("Base URL", value="https://api.openai.com/v1")
        model_name = st.sidebar.text_input("Model Name", value="gpt-4o")

        st.sidebar.info(
            "You can use any API that is compatible with the OpenAI API standard. "
            "Please enter your API Key, Base URL, and Model Name accordingly."
        )

        st.divider()

        if st.button("Clear Chat"):
            st.session_state.clear()
            await checkpointer.adelete_thread(THREAD_ID)

    # Chatbox initialization
    chatbox = ChatBox(use_rich_markdown=True)
    chatbox.output_messages()

    state = State(
        user_input="",
        messages=[],
        model_config=OpenAIModelConfig(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
        ),
    )

    config = RunnableConfig(
        configurable={"thread_id": THREAD_ID},
        callbacks=[CallbackHandler()] if langfuse.auth_check() else [],
    )

    if query := st.chat_input(
        "input your question here", disabled=not api_key or not model_name
    ):
        chatbox.user_say(query)
        state["user_input"] = query

        chatbox.ai_say([])
        async for chunk in graph.astream(state, config=config, stream_mode="custom"):
            if isinstance(chunk, StreamMessage):
                if chunk.type == "tool_call_start":
                    chatbox.insert_msg(
                        Markdown(
                            f"Tool call {chunk.tool_call_name or 'N/A'} started with args {chunk.tool_call_args }",
                            title="Tool call",
                            in_expander=True,
                            expanded=True,
                        )
                    )
                    chatbox.update_msg(
                        "\n\n", element_index=0, streaming=False, state="complete"  # type: ignore
                    )
                elif chunk.type == "tool_call_end":
                    # print(chunk)
                    chatbox.update_msg(
                        f"Result: {chunk.content or 'N/A'}",
                        streaming=False,
                        state="complete",  # type: ignore
                    )
                elif chunk.type == "text":
                    # If chatbox last message is not text insert text else update last message
                    try:
                        last_markdown = (
                            chatbox.history[-1].get("elements")[-1].to_dict()
                        )
                    except Exception:
                        last_markdown = None

                    if last_markdown is None or last_markdown.get("title") != "text":
                        chatbox.insert_msg(Markdown(chunk.content or "", title="text"))
                    else:
                        chatbox.update_msg(chunk.content or "", streaming=True)
                elif chunk.type == "thinking":
                    try:
                        last_markdown = (
                            chatbox.history[-1]
                            .get("elements")[-1]
                            .to_dict()
                            .get("thinking")
                        )
                    except Exception:
                        last_markdown = None

                    if (
                        last_markdown is None
                        or last_markdown.get("title") != "thinking"
                    ):
                        chatbox.insert_msg(
                            Markdown(chunk.content or "", title="thinking")
                        )
                    else:
                        chatbox.update_msg(
                            (last_markdown.get("content") or "")
                            + (chunk.content_delta or ""),
                            streaming=True,
                        )


if __name__ == "__main__":
    asyncio.run(render_streamlit())
