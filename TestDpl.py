import streamlit as st
import json
import os
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import END

# Load API Key
os.environ["OPENAI_API_KEY"] = "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"

# Load dữ liệu JSON
with open("products_detail.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

# Chuyển dữ liệu thành dạng Document
chunks = [json.dumps(item, ensure_ascii=False, indent=2) for item in json_data]
documents = [Document(page_content=chunk) for chunk in chunks]

# Tạo Embeddings và VectorStore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
vector_store.add_documents(documents)

# Khởi tạo LLM (GPT-4o-mini)
llm = init_chat_model("gpt-4o-mini", model_provider="openai")


# Công cụ tìm kiếm dữ liệu
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Truy vấn dữ liệu sản phẩm liên quan đến câu hỏi."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Xây dựng chatbot với LangGraph
graph_builder = StateGraph(MessagesState)


def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tools = ToolNode([retrieve])


def generate(state: MessagesState):
    recent_tool_messages = [msg for msg in reversed(state["messages"]) if msg.type == "tool"]
    docs_content = "\n\n".join(doc.content for doc in recent_tool_messages[::-1])

    system_message_content = (
        "Bạn là trợ lý cửa hàng chuyên tư vấn sản phẩm. Hãy chỉ trả lời dựa trên thông tin được cung cấp."
        f"\n\n{docs_content}"
    )
    prompt = [SystemMessage(system_message_content)] + [
        msg for msg in state["messages"] if msg.type in ("human", "system")
    ]

    response = llm.invoke(prompt)
    return {"messages": [response]}


# Cấu hình LangGraph
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)
graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges("query_or_respond", tools_condition, {END: END, "tools": "tools"})
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Giao diện Streamlit
st.set_page_config(page_title="Chatbot tư vấn sản phẩm", page_icon="🛍️")
st.title("🛍️ Chatbot Tư Vấn Sản Phẩm")

# Lưu trạng thái hội thoại
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Hiển thị hội thoại
for msg in st.session_state["messages"]:
    role = "🤖 Chatbot" if msg["role"] == "assistant" else "🧑 Bạn"
    st.chat_message(role).markdown(msg["content"])

# Ô nhập tin nhắn
user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    config = {"configurable": {"thread_id": "session_1"}}
    responses = []

    for step in graph.stream({"messages": [{"role": "user", "content": user_input}]}, stream_mode="values",
                             config=config):
        responses.append(step["messages"][-1].content)

    chatbot_response = responses[-1]
    st.session_state["messages"].append({"role": "assistant", "content": chatbot_response})

    st.chat_message("🤖 Chatbot").markdown(chatbot_response)
