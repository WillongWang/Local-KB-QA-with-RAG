import time
import requests

import os
import gradio as gr
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

import torch
torch.cuda.empty_cache()

'''
import pandas as pd
import os

file_path = 'documents/jd_faq.csv'

if os.path.exists(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 计算每行的字符总数
    row_lengths = df.astype(str).apply(lambda x: x.str.len().sum(), axis=1)
    
    # 显示最大行长度
    max_length = row_lengths.max()
    max_index = row_lengths.idxmax()
    
    print(f"最大行长度: {max_length} 字符，位于行号: {max_index}")
    print(f"该行内容:\n{df.iloc[max_index]}")
    
    # 查看所有行的长度统计
    print(f"行长度统计:\n最小值: {row_lengths.min()}\n平均值: {row_lengths.mean()}\n中位数: {row_lengths.median()}")
    print(f"前5行长度: {row_lengths.head(5).tolist()}")
else:
    print(f"文件 {file_path} 不存在")
'''

'''
def chat(prompt,history=None):
    payload={
        "prompt":prompt,"history":[] if not history else history
    }
    headers={"Content-Type":"application/json"}
    resp = requests.post(
        url='http://127.0.0.1:8000',
        json=payload,
        headers=headers)#{"Content-Type": "application/json;charset=utf-8"}
    return resp.json()['response'],resp.json()['history']

history = []

while True:
    response, history = chat(input("Question:"), history)
    print('Answer:', response)
'''    
# while True:
#  query = input("Human: ")
#  similar_docs = db.similarity_search(query, include_metadata=True, k=4) #db见后
#  prompt="基于下面给出的资料，回答问题。如果资料不足，回答不了，就回复不知道。下面是资料：\n"
#  for idx,doc in enumerate(similar_docs):
#   prompt+=f"{idx+1}：{doc.page_content}\n"
#  prompt+=f"下面是问题：{query}"
#  response,_=chat(prompt,[])
#  print("Bot：",response)


#加载embedding
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec2": "uer/sbert-base-chinese-nli",
    "text2vec3": "shibing624/text2vec-base-chinese",
    #"bge-small-zh-v1": "BAAI/bge-small-zh-v1",
    #"all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2"
}

import pandas as pd
from langchain.schema import Document
def load_documents_csv(file_path="documents/jd_faq.csv"):
    """ 
    :param directory:
    :return:
    """
    documents = []
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        for index, row in df.iterrows():
            # 根据CSV结构调整这里
            question = row.get("问题", "")
            answer = row.get("答案", "")
            category = row.get("类别", "")
            subcategory = row.get("子类别", "") 
            # 合并问题和答案作为文档内容
            content = f"类别: {category}\n子类别: {subcategory}\n问题: {question}\n答案: {answer}"   
            # 创建文档对象
            doc = Document(
                page_content=content,
                metadata={"source": f"row_{index}", "category": category}
            )
            documents.append(doc)    
    return documents

def load_documents(directory="documents\\books"): #只上传txt文件
    #loader = TextLoader(directory,encoding='utf-8') 老是错
    loader = DirectoryLoader(directory)
    documents = loader.load()
    text_spliter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    split_docs = text_spliter.split_documents(documents)
    #print(split_docs) #
    return split_docs

def load_embedding_model(model_name="ernie-tiny"):
    """
    加载embedding模型
    :param model_name:
    :return:
    """
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"}
    return HuggingFaceEmbeddings(
        model_name=embedding_model_dict[model_name],
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

# 加载embedding模型
embeddings = load_embedding_model('text2vec3')

if not os.path.exists('VectorStore'): # 只有首次运行会执行这个代码块
    documents = load_documents_csv()  
    db = Chroma.from_documents(documents, embeddings,persist_directory='VectorStore')
else:
    db = Chroma(persist_directory='VectorStore', embedding_function=embeddings)
    
# 加载数据库
def store_chroma(split_docs):
    """
    将文档向量化，存入向量数据库
    :param split_docs:
    :return:
    """
    try:
        db.add_documents(split_docs)
        #db.persist() 出错
        print(f"成功处理文件")
    except Exception as e:
        print(f"文件处理失败")
#创建llm
#from langchain_community.llms import ChatGLM
# llm = ChatGLM(
#     endpoint_url='http://127.0.0.1:8000',
#     max_token=80000,
#     top_p=0.9
# )
from langchain_deepseek import ChatDeepSeek
llm=ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=2000,
    api_key=""
)

# 创建qa
QA_CHAIN_PROMPT = PromptTemplate.from_template("""根据以下已知信息回答问题：
{context}
问题：{question}
请用中文简洁回答，如果无法回答请说不知道。""")
retriever = db.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            verbose=True,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
# response=qa.run("注册企业会员需要收费吗")  #"渡边是谁"
# print(response)
# torch.cuda.empty_cache()
# def chat(question, history):
#  response=qa.run(question)
#  torch.cuda.empty_cache()
#  return response
# demo=gr.ChatInterface(chat)
# demo.launch(inbrowser=True)

'''
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("E:\ChatGLM-6B-main\self-chatglm", trust_remote_code=True)
model = AutoModel.from_pretrained("E:\ChatGLM-6B-main\self-chatglm", trust_remote_code=True).half().cuda()
model.eval()
response,history=model.chat(tokenizer,"你好",history=[])
print(response)
response,history=model.chat(tokenizer,"晚上睡不着应该怎么办",history=history)
print(response)
'''

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    if message["files"] is not None:
        for x in message["files"]:
            print("..."+x) #
            parent = x.rsplit('\\', 1)[0] #只支持文件下的文档上传，且会把该文件下的全向量化
            documents = load_documents(parent)
            store_chroma(documents)
            history.append({"role": "user", "content": {"path": x}})
        retriever = db.as_retriever(search_kwargs={"k": 3})
        global qa
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            verbose=True,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
    if message["text"] is not None:
        print("..."+message["text"]) #
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot(history: list):
    print("...."+history[-1]["content"]) #
    result = qa({"query": history[-1]["content"]})
    print("答案：", result["result"])
    print("\n=== 支持证据 ===")
    for doc in result["source_documents"]:
        print(f"来源：{doc.metadata['source']}")
        print(f"内容：{doc.page_content}\n{'-'*50}")
    history.append({"role": "assistant", "content": ""})
    for character in result["result"]:
        history[-1]["content"] += character
        time.sleep(0.05)
        yield history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages")

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Enter message or upload file...",
        show_label=False,
        sources=["microphone", "upload"],
    )

    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], [chatbot, chat_input]
    )
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    chatbot.like(print_like_dislike, None, None, like_user_message=True)

demo.launch(inbrowser=True)
