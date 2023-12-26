import streamlit as st
import torch
import copy
from transformers import AutoModel, AutoTokenizer

in_response=[] #多个内部回答

# 设置页面标题、图标和布局
st.set_page_config(
    page_title="ChatGLM3-Ziwei 演示",
    page_icon=":robot:",
    layout="wide"
)

# 设置为模型ID或本地文件夹路径
model_path = "/home/admin1/桌面/chatglm3-6b"

@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()
    # 多显卡支持,使用下面两行代替上面一行,将num_gpus改为你实际的显卡数量
    # from utils import load_model_on_gpus
    # model = load_model_on_gpus("THUDM/chatglm3-6b", num_gpus=2)
    model = model.eval()
    return tokenizer, model

# 加载Chatglm3的model和tokenizer
tokenizer, model = get_model()

# 初始化历史记录和past key values
if "history" not in st.session_state:
    st.session_state.history = []
if "past_key_values" not in st.session_state:
    st.session_state.past_key_values = None

# 设置max_length、top_p和temperature
max_length = st.sidebar.slider("max_length", 0, 32768, 8192, step=1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.6, step=0.01)

# 清理会话历史
buttonClean = st.sidebar.button("清理会话历史", key="clean")
if buttonClean:
    st.session_state.history = []
    st.session_state.past_key_values = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.rerun()

# 渲染聊天历史记录
for i, message in enumerate(st.session_state.history):
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            st.markdown(message["content"])
    else:
        with st.chat_message(name="assistant", avatar="assistant"):
            st.markdown(message["content"])

# 输入框和输出框
with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
with st.chat_message(name="assistant", avatar="assistant"):
    message_placeholder = st.empty()

#prompt函数
def template_change(template,a,b,i):
    template1=template.replace(a,b,i)
    return template1

def get_classify(question,history,past_key_values):
    template_string = """
    请判断下列问题属于占卜的哪一种分类或主题
    注意：你只需要输出你判断的主题分类，即输出一个或几个词语，而不是一段话。
    ###问题:{question}
    """
    # 填充变量
    prompt = template_change(template_string,'{question}',question,1)
    #print('prompt 1： '+prompt)
    i=0
    for classify, history, past_key_values in model.stream_chat(
            tokenizer,
            prompt,
            history,
            past_key_values=past_key_values,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            return_past_key_values=True,
    ):
        i+=1
    print("1:  " + classify)

    return classify


# 多问题回答函数
def prompt_main(question, history,past_key_values, theme,num):
    global in_response
    # 定义模板字符串
    ####主题：{theme}
    template_string = """
    你现在是一位占卜师，你需要根据下面的问题与我对话，回答需要解释答案。
    ###问题: {question}
    对话主题：{theme}
    """
    # 如果问题的答案需要询问者提供信息，那么不要捏造信息，询问相关的信息。
    # 使用模板字符串创建一个提示模板
    # 填充变量             ,theme=new_question
    prompt0 = template_change(template_string,'{question}', question,1)
    prompt=template_change(prompt0,'{theme}', theme,1)

    i=0
    for inresponse, history, past_key_values in model.stream_chat(
        tokenizer,
        prompt,
        history,
        past_key_values=past_key_values,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature,
        return_past_key_values=True,
    ):
        i+=1
    in_response.append(inresponse)
    print("2:" + in_response[num])

# 多回答合并函数
def prompt_merge(num, question,history, past_key_values):
    global in_response
    reply = ''
    for i in range(num):
        reply += '\n第' + str(i + 1) + '段文字:   '
        reply += in_response[i]
    #print('reply ： ' + reply)
    # 定义模板字符串
    template_string = """
    请把下面{num}段文字改写合并为一段流畅的文字
    回答时开头不要出现【改写后的文字如下：】
    整合后的文字不能重复出现相似的内容
    整合后的文字应该尽量包含{num}段文字里不同的内容
    ###{reply} 
    """
    # 填充变量
    prompt0 = template_change(template_string,'{num}', str(num),2)
    prompt=template_change(prompt0,'{reply}',reply,1)
    #print('prompt 3： ' + prompt)
    i=0
    for out_response ,history,past_key_values in model.stream_chat(
            tokenizer,
            prompt,
            history,
            past_key_values=past_key_values,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            return_past_key_values=True,
    ):
        message_placeholder.markdown(out_response)
    print("3:" + out_response)
    # for i in range(len(history)):
    #     print("history  ",history[i])
    return history[-1]

# 获取用户输入
prompt_text = st.chat_input("请输入您的问题")

# 如果用户输入了内容,则生成回复
if prompt_text:
    input_placeholder.markdown(prompt_text)
    history = st.session_state.history
    past_key_values = st.session_state.past_key_values
    history1=copy.deepcopy(st.session_state.history)
    past_key_values1 = st.session_state.past_key_values

    num=4
    theme = get_classify(prompt_text,history,past_key_values)
    history = copy.deepcopy(history1)
    for i in range(num):
        prompt_main(prompt_text,history,past_key_values, theme, i)
        history = copy.deepcopy(history1)
    history.append(prompt_merge(num, prompt_text,history,past_key_values))
    # for i in range(len(history)):
    #     print("history1  ",history[i])
    h=[]
    i=0
    for response, h, past_key_values in model.stream_chat(
        tokenizer,
        prompt_text,
        h,
        past_key_values=past_key_values,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature,
        return_past_key_values=True,
    ):
        i+=1
    history[-2]=h[-2]
    # for i in range(len(history)):
    #     print('history： ' , history[i])


    # for response, history, past_key_values in model.stream_chat(
    #     tokenizer,
    #     prompt_text,
    #     history,
    #     past_key_values=past_key_values,
    #     max_length=max_length,
    #     top_p=top_p,
    #     temperature=temperature,
    #     return_past_key_values=True,
    # ):
    #     message_placeholder.markdown(response)

    # 更新历史记录和past key values
    st.session_state.history = history
    st.session_state.past_key_values = past_key_values
# streamlit run prompt_web.py
#  请你给我算算今年运势
