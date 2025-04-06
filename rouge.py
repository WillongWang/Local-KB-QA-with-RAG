from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import os
import pandas as pd

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

from langchain_deepseek import ChatDeepSeek
llm=ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=8000, # 大于8000会报错
# jd_faq.csv record的最大行长度: 15290 字符，位于行号: 415
# 行长度统计:
# 最小值: 17
# 平均值: 353.41143654114364
# 中位数: 132.0
    api_key=""
)

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
    response = qa.run(question)
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

# llm有时会在生成的最前面加上“答案：”，所以需要去除
def clean_answer(text):
    if text.startswith('答案:'):
        return text[3:]
    return text
hyp_processed = [preprocess_cn(clean_answer(t)) for t in generated_answers]
ref_processed = [preprocess_cn(t) for t in original_answers]
scores = rouge.get_scores(hyp_processed, ref_processed, avg=True)
print("ROUGE__ scores:", scores)

'''
ROUGE scores: {'rouge-1': {'r': 0.8697536961316201, 'p': 0.8820212403740397, 'f': 0.8622482535036113}, 'rouge-2': {'r': 0.8572443412976866, 'p': 0.8492444942433046, 'f': 0.8484520163364817}, 'rouge-l': {'r': 0.8635680637155919, 'p': 0.8766723403198635, 'f': 0.8559542361842964}}
ROUGE__ scores: {'rouge-1': {'r': 0.869738369756461, 'p': 0.8924063052186957, 'f': 0.8677424183856427}, 'rouge-2': {'r': 0.8572443412976866, 'p': 0.8587797806030869, 'f': 0.8537843303869588}, 'rouge-l': {'r': 0.8635680637155919, 'p': 0.8852282704372529, 'f': 0.8605232161156948}}
'''
