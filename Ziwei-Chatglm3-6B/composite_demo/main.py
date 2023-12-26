import streamlit as st
st.set_page_config(
    page_title="ChatGLM3-Ziwei Demo",
    page_icon=":robot:",
    layout='centered',
    initial_sidebar_state='expanded',
)


import demo_chat
from enum import Enum

DEFAULT_SYSTEM_PROMPT = '''
You are ChatGLM3-Ziwei, a large language model trained by HITWH. Follow the user's instructions carefully. Respond using markdown.
'''.strip()

# Set the title of the demo
st.title("ChatGLM3-Ziwei Demo")

# Add your custom text here, with smaller font size


class Mode(str, Enum):
    CHAT= '💬 Chat'


with st.sidebar:
    top_p = st.slider(
        'top_p', 0.0, 1.0, 0.8, step=0.01
    )
    temperature = st.slider(
        'temperature', 0.0, 1.5, 0.95, step=0.01
    )
    repetition_penalty = st.slider(
        'repetition_penalty', 0.0, 2.0, 1.1, step=0.01
    )
    max_new_token = st.slider(
        'Output length', 5, 32000, 256, step=1
    )

    cols = st.columns(2)
    export_btn = cols[0]
    clear_history = cols[1].button("Clear History", use_container_width=True)
    retry = export_btn.button("Retry", use_container_width=True)

    system_prompt = st.text_area(
        label="System Prompt (Only for chat mode)",
        height=300,
        value=DEFAULT_SYSTEM_PROMPT,
    )

prompt_text = st.chat_input(
    'Chat with ChatGLM3!',
    key='chat_input',
)

tab = st.radio(
    'Mode',
    [mode.value for mode in Mode],
    horizontal=True,
    label_visibility='hidden',
)

if clear_history or retry:
    prompt_text = ""

match tab:
    case Mode.CHAT:
        demo_chat.main(
            retry=retry,
            top_p=top_p,
            temperature=temperature,
            prompt_text=prompt_text,
            system_prompt=system_prompt,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_token
        )

    case _:
        st.error(f'Unexpected tab: {tab}')
