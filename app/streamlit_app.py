"""Streamlit chat UI prototype for Yokohama architectural regulations."""

from html import escape

import streamlit as st

from mock_data import get_mock_response


st.set_page_config(
    page_title="横浜市 建築規制 RAG",
    page_icon="🏙️",
    layout="centered",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    :root {
        --app-text: #202123;
        --app-muted: #6b7280;
        --app-border: #e5e7eb;
        --app-soft: #f7f7f8;
    }

    .stApp {
        color: var(--app-text);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 780px;
        padding-top: 2.1rem;
        padding-bottom: 7rem;
    }

    .app-header {
        margin-bottom: 2rem;
    }

    .app-title {
        font-size: 1.35rem;
        font-weight: 650;
        letter-spacing: -0.01em;
        margin: 0;
    }

    .app-subtitle {
        color: var(--app-muted);
        font-size: 0.9rem;
        margin-top: 0.35rem;
    }

    .welcome-state {
        text-align: center;
        padding: 12vh 0 2.2rem;
    }

    .welcome-state h1 {
        font-size: clamp(1.65rem, 4vw, 2.15rem);
        font-weight: 600;
        line-height: 1.4;
        letter-spacing: -0.02em;
        margin: 0;
    }

    [data-testid="stChatMessage"] {
        padding: 1rem 0;
        background: transparent;
    }

    [data-testid="stChatMessageContent"] p {
        line-height: 1.8;
    }

    [data-testid="stChatInput"] {
        border: 1px solid var(--app-border);
        border-radius: 1.25rem;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.06);
    }

    [data-testid="stExpander"] {
        border-color: var(--app-border);
        border-radius: 0.75rem;
        margin-top: 1rem;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--app-border);
    }

    .sidebar-label {
        color: var(--app-muted);
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin: 1.3rem 0 0.45rem;
        text-transform: uppercase;
    }

    .history-item {
        color: #374151;
        font-size: 0.9rem;
        overflow: hidden;
        padding: 0.42rem 0;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .sidebar-spacer {
        min-height: 12vh;
    }

    .sidebar-info {
        border-top: 1px solid var(--app-border);
        color: var(--app-muted);
        font-size: 0.78rem;
        line-height: 1.7;
        padding-top: 1rem;
    }

    .sidebar-info strong {
        color: #4b5563;
        font-weight: 600;
    }

    @media (max-width: 720px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1.2rem;
        }

        .welcome-state {
            padding-top: 7vh;
        }
    }
</style>
"""


EXAMPLE_QUESTIONS = [
    "綱島C地区の高さ制限は？",
    "建ぺい率の最高限度は？",
    "道路から何m後退する必要がありますか？",
]


def initialize_session_state() -> None:
    """Initialize state once so messages survive Streamlit reruns."""

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []


def render_assistant_result(result: dict) -> None:
    """Render a structured backend-style result inside an assistant message."""

    st.markdown(f"**{result['answer']}**")

    st.markdown("### 根拠となる規定")
    st.write(result["regulation"])

    st.markdown("### 適用条件")
    st.write(result["conditions"])

    st.markdown("### 注意事項")
    st.write(result["notes"])

    source = result["source"]
    st.markdown("### 出典")
    st.write(source["title"])
    st.caption(f"地区計画番号：{source['plan_id']}")
    st.markdown(f"[原文を見る ↗]({source['url']})")

    retrieved_chunks = result.get("retrieved_chunks", [])
    with st.expander("参照した資料を見る", expanded=False):
        if not retrieved_chunks:
            st.caption("このモック回答には参照チャンクがありません。")
            return

        for index, chunk in enumerate(retrieved_chunks, start=1):
            st.markdown(f"**参照 {index}**")
            st.markdown(f"**資料名：**  \n{chunk['document']}")
            if chunk.get("district"):
                st.markdown(f"**地区：**  \n{chunk['district']}")
            st.markdown(f"**項目：**  \n{chunk['section']}")
            st.markdown(f"**本文：**  \n{chunk['text']}")
            if index < len(retrieved_chunks):
                st.divider()


def render_sidebar() -> None:
    """Render new-chat control, lightweight history, and corpus metadata."""

    with st.sidebar:
        if st.button("＋ 新しいチャット", width="stretch"):
            st.session_state.messages = []
            st.rerun()

        st.markdown('<div class="sidebar-label">履歴</div>', unsafe_allow_html=True)
        if st.session_state.history:
            for item in reversed(st.session_state.history[-8:]):
                st.markdown(
                    f'<div class="history-item">{escape(item)}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("まだ履歴はありません")

        st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sidebar-info">
                <strong>対象自治体</strong><br>横浜市<br><br>
                <strong>検索対象</strong><br>地区計画<br><br>
                <strong>Embedding</strong><br>Ruri v3 130M<br><br>
                <strong>Top-K</strong><br>3
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_conversation() -> None:
    """Replay all current messages in conversation order."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                render_assistant_result(message["result"])


def submit_question(question: str) -> None:
    """Append and render one mock user/assistant exchange."""

    question = question.strip()
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    if question not in st.session_state.history:
        st.session_state.history.append(question)

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("関連する規定を検索しています..."):
            result = get_mock_response(question)
        render_assistant_result(result)

    st.session_state.messages.append({"role": "assistant", "result": result})


initialize_session_state()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
render_sidebar()

st.markdown(
    """
    <div class="app-header">
        <p class="app-title">横浜市 建築規制 RAG</p>
        <p class="app-subtitle">横浜市の地区計画・建築規制について質問できます</p>
    </div>
    """,
    unsafe_allow_html=True,
)

render_conversation()

submitted_question = None
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-state">
            <h1>横浜市の建築規制について<br>何を調べますか？</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for index, example_question in enumerate(EXAMPLE_QUESTIONS):
        if st.button(
            example_question,
            key=f"example-question-{index}",
            width="stretch",
        ):
            submitted_question = example_question

chat_question = st.chat_input("横浜市の地区計画について質問する")
if chat_question:
    submitted_question = chat_question

if submitted_question:
    submit_question(submitted_question)
    st.rerun()
