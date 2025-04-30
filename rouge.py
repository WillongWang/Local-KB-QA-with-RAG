from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import pandas as pd
import openai

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec2": "uer/sbert-base-chinese-nli",
    "text2vec3": "shibing624/text2vec-base-chinese",
    #"bge-small-zh-v1": "BAAI/bge-small-zh-v1",
    #"all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2"
}

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

def load_embedding_model(model_name="ernie-tiny"):
    """
    加载embedding模型
    :param model_name:
    :return:
    """
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cpu"} # "cuda:0"
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

from langchain_deepseek import ChatDeepSeek
llm=ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=8000, # 大于8000会报错
# jd_faq.csv record的最大行长度: 15290 字符，位于行号: 415
# 行长度统计:
# 最小值: 17
# 平均值: 353.41143654114364
# 中位数: 132.0
    api_key=""
)
'''
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    temperature=0,
    model="qwen-max", # glm-4-plus
    openai_api_key="",
    openai_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", # https://open.bigmodel.cn/api/paas/v4
    max_retries=100,
    timeout=1000
)
'''
# qwen-max
def safe_qa_run(qa, question):
    try:
        response = qa.run(question)
        return response
    except openai.BadRequestError as e:
        print(f"Skipping question due to content inspection failure: {question}")
        return "无"  # or return a default response
    except Exception as e:
        print(f"Unexpected error for question: {question}")
        print(f"Error: {e}")
        return "无"
# > Entering new RetrievalQA chain...
# Skipping question due to content inspection failure: 图书、音乐、影视、教育类商品退换货细则（三方）
# 450

# 读取原始数据
df = pd.read_csv("documents/jd_faq.csv")
original_answers = df['答案'].tolist()
questions = df['问题'].tolist()

QA_CHAIN_PROMPT = PromptTemplate.from_template("""根据以下已知信息回答问题：
{context}
问题：{question}
请用原文回答，如果无法回答请说不知道。""")
retriever = db.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            verbose=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

# 生成预测答案
generated_answers = []
i=0
for question in questions:
    response = safe_qa_run(qa, question)
    generated_answers.append(response.strip())
    print(i)
    i+=1

# 保存结果
result_df = pd.DataFrame({
    '问题': questions,
    '原答案': original_answers,
    '生成答案': generated_answers
})
result_df.to_csv('comparison_results.csv', index=False)

df = pd.read_csv("comparison_results.csv")
original_answers = df['原答案'].tolist()
generated_answers = df['生成答案'].tolist()
questions = df['问题'].tolist()

import jieba
from rouge_chinese import Rouge

def preprocess_cn(text):
    # 中文分词+去除空格
    return ' '.join(jieba.cut(text.strip()))

# 预处理文本
hyp_processed = [preprocess_cn(t) for t in generated_answers]
ref_processed = [preprocess_cn(t) for t in original_answers]

rouge = Rouge()
scores = rouge.get_scores(hyp_processed, ref_processed, avg=True)
print("ROUGE scores:", scores)

# llm有时会加上“答案”，所以需要去除
str = "答案" # "答案如下","答案是","回答"
def clean_answer(str,id,text):
    if str in text:
        if text.split(str)[0]!="":
            print("问题："+questions[id]+f"\n'{str}'前："+text.split(str)[0].strip()+f"\n'{str}'后："+text[text.find(str)+2:].strip())
        return text[text.find(str)+3:].strip() if text[text.find(str)+2:].strip().startswith((":", "：", ".", "。", ")", "）")) else text[text.find(str)+2:].strip()
    return text
hyp_processed = [preprocess_cn(clean_answer(str,id,t)) for id,t in enumerate(generated_answers)]
ref_processed = [preprocess_cn(t) for t in original_answers]
scores = rouge.get_scores(hyp_processed, ref_processed, avg=True)
print("ROUGE__ scores:", scores)

i = 0
for t in generated_answers:
    if '不知道' in t: # '抱歉'
        print("不知道："+t.strip())
        i = i+1
print(f"error: {i/len(generated_answers)*100}%")


'''
deepseek:
ROUGE scores: {'rouge-1': {'r': 0.8760813716181665, 'p': 0.8876677427251719, 'f': 0.8688721773393938}, 'rouge-2': {'r': 0.8629701285016342, 'p': 0.8563972453435853, 'f': 0.8548255888685075}, 'rouge-l': {'r': 0.8698783574488914, 'p': 0.8820809624132675, 'f': 0.8624301011466202}}
ROUGE__ scores: {'rouge-1': {'r': 0.8754581772148489, 'p': 0.902772666296256, 'f': 0.8767105362920002}, 'rouge-2': {'r': 0.8628527559920267, 'p': 0.870267986627958, 'f': 0.8632608376814965}, 'rouge-l': {'r': 0.86937738106516, 'p': 0.8955599694644488, 'f': 0.8693496377915781}}
error: 7.531380753138076%
glm-4-plus:
ROUGE scores: {'rouge-1': {'r': 0.8518859914825461, 'p': 0.874679795128485, 'f': 0.8467874580183553}, 'rouge-2': {'r': 0.8424342688647568, 'p': 0.8426113624384427, 'f': 0.8350328027401857}, 'rouge-l': {'r': 0.8468614997304648, 'p': 0.8676560489060653, 'f': 0.8393518956063892}}
ROUGE__ scores: {'rouge-1': {'r': 0.8518150054267527, 'p': 0.8752705433449542, 'f': 0.8469820194714002}, 'rouge-2': {'r': 0.8417269383862235, 'p': 0.8431981990062315, 'f': 0.8353223588571559}, 'rouge-l': {'r': 0.846808794814738, 'p': 0.8682616084095186, 'f': 0.8396140210290138}}
error: 11.157601115760112%
qwen-max:
ROUGE scores: {'rouge-1': {'r': 0.8329327940549331, 'p': 0.8948467787706476, 'f': 0.8353624825072422}, 'rouge-2': {'r': 0.827109130654462, 'p': 0.8346998768854681, 'f': 0.8270371516715648}, 'rouge-l': {'r': 0.8307735815271367, 'p': 0.8928177887917117, 'f': 0.8317843659761307}}
ROUGE__ scores: {'rouge-1': {'r': 0.832374913999145, 'p': 0.8967940973354741, 'f': 0.8365805989996233}, 'rouge-2': {'r': 0.8259933705428861, 'p': 0.8366625448794439, 'f': 0.8283590435969906}, 'rouge-l': {'r': 0.8301758528959353, 'p': 0.8947152365768737, 'f': 0.8330658558117658}}
error: 14.923291492329149%
'''
