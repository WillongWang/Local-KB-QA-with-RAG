# Enhancing Local Knowledge Base QA with LLMs and RAG Integration

This project aims to enhance local knowledge base QA in Chinese by integrating **Chinese LLMs** with **RAG** technology using **LangChain**. It enables users to upload unstructured documents and generates accurate, context-aware responses through similarity search. A **chatbot GUI** is built with **Gradio** to provide an intuitive user interface for interacting with the system. The project plans to evaluate various prompt strategies on the JD QA dataset and some literary texts using precision and ROUGE metrics for performance assessment.

The project follows the fundamental RAG pipeline, which involves several key steps. First, the uploaded documents are split into smaller chunks based on predefined rules, making them easier to process. Next, these chunks are embedded and vectorized to facilitate similarity comparison. When a user submits a query, the system performs a similarity search on the embedded document vectors to identify the most relevant passages. The retrieved content is then combined with the user’s query to generate a coherent and accurate response.

## Chinese LLMs

1.CHATGLM

1) Deploy locally

Deploy ChatGLM-6B according to the [official website](https://github.com/THUDM/ChatGLM-6B) (included in this project). I deployed it locally on Windows using the API and directly loaded the quantized model. The INT4 quantized model only requires about 5.2GB of memory. Model quantization may lead to some performance loss, but through testing, ChatGLM-6B can still generate natural and fluent responses with 4-bit quantization. You are recommended to download the model directly from Hugging Face ([chatglm-6b](https://huggingface.co/THUDM/chatglm-6b/tree/main) or [chatglm-6b-int4](https://huggingface.co/THUDM/chatglm-6b-int4/tree/main)) and remember to install the GCC for the INT4 model.

```
# Replace the corresponding part in the original official api.py, where self-chatglm-6b-int4 is your local chatglm-6b or chatglm-6b-int4 directory
tokenizer = AutoTokenizer.from_pretrained("self-chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("self-chatglm-6b-int4", trust_remote_code=True).half().cuda()
model.eval()  
```

Then, deploy the ChatGLM model:

```
# first create a separate virtual environment for deploying
pip install -r requirements_chatglm.txt
python api.py
```

Create the LLM using the following approach in chatglm_document_qa.py:

```
from langchain_community.llms import ChatGLM

llm = ChatGLM(
     endpoint_url='http://127.0.0.1:8000',
     max_token=80000,
     top_p=0.9)
```

2) API (state-of-the-art GLM-4 series model **glm-4-plus**)

```
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    temperature=0,
    model="glm-4-plus", # qwen-max
    openai_api_key="",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4", # https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    max_retries=100,
    timeout=1000
)
```

I also tried state-of-the-art Qwen series model **qwen-max** to compare performances of these Chinese LLMs as mentioned later.

The performance of **gpt-35-turbo** in this QA is the worst, which might be due to the lack of optimization for handling Chinese:

```
import getpass
from langchain_openai import AzureChatOpenAI
if "AZURE_OPENAI_API_KEY" not in os.environ:
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass(
        "Enter your AzureOpenAI API key: "
    )
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://hkust.azure-api.net" # replace
llm = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",  # or your deployment
    api_version="2023-05-15",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=10,
    # other params...
)
```

2.DeepSeek

To balance time and cost, I also used the DeepSeek API. According to the official website, the deepseek-chat model has been fully upgraded to DeepSeek-V3.

```
from langchain_deepseek import ChatDeepSeek

llm=ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=2000,
    api_key="") #Your api_key
```

## How to run

```
pip install -r requirements.txt
python chatglm_document_qa.py
```

Default settings:

You can adjust them accordingly.

```
# chunk_size & chunk_overlap to split the original or uploaded text
text_spliter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0) 

# embedding
embedding_model_dict = { 
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec2": "uer/sbert-base-chinese-nli",
    "text2vec3": "shibing624/text2vec-base-chinese",
    #"bge-small-zh-v1": "BAAI/bge-small-zh-v1",
    #"all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2"
}
embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_dict["text2vec3"],
        #......
    )

# prompt
QA_CHAIN_PROMPT = PromptTemplate.from_template("""根据以下已知信息回答问题：
{context}
问题：{question}
请用中文简洁回答，如果无法回答请说不知道。""")

retriever = db.as_retriever(search_kwargs={"k": 3}) # return 3 relevant docs
```

This file by default processes a small JD Q&A dataset [jd_faq.csv](https://github.com/WillongWang/Local-KB-QA-with-RAG/blob/main/documents/jd_faq.csv) as an experiment.  
Each line is used to construct a langchain.schema.Document object and is then vectorized and stored. You can comment out the corresponding code snippet if needed.

### ROUGE & [BERTSCORE](https://arxiv.org/pdf/1904.09675)

This project uses ROUGE and BERTSCORE to evaluate the QA performance. Each record in [jd_faq.csv](https://github.com/WillongWang/Local-KB-QA-with-RAG/blob/main/documents/jd_faq.csv) contains a question and its corresponding answer. The LLM is prompted to answer each question, and the generated answer is compared with the original one to compute the scores.

```
# pip install jieba rouge_chinese bert_score
python rouge.py
python bertscore.py
```

#### Necessary adjustments and unavoidable limitations:

1. The maximum supportable number of tokens generated by DeepSeek is smaller than the maximum length of answers in the dataset.

```
# The maximum record (line) length in jd_faq.csv: 15,290 characters, at line number 415
# Record length statistics:
# Minimum: 17
# Average: 353.41
# Median: 132.0
llm = ChatDeepSeek( 
    model="deepseek-chat",
    temperature=0,
    max_tokens=8000,  # Setting more than 8000 will result in an error
    api_key=""
)
```

Actually, for the question with the longest answer (line 415: "国际机票中国国际航空旅客、行李运输须知"), the model generated the following response:  
"国际机票中国国际航空旅客、行李运输须知的原文内容如下： ...(original text) （后续条款因篇幅限制未完整列出，完整内容请参考中国国际航空官方文件。）
如需具体某一条款的详细内容，请告知具体条款编号。"  
This showcases the DeepSeek’s strong QA capabilities — it correctly generated an appropriate introduction and thoughtfully indicated that some clauses were omitted due to length. However, since ROUGE-N precision mainly captures surface-level word overlap, it may underrate the model’s true ability and fail to reflect its actual understanding and reasoning.  
However, glm-4-plus and qwen-max failed to answer this.

2. To make the LLM answer more accurately, in addition to set **all temperatures to 0**, it is explicitly instructed to respond based on the provided context:

```
QA_CHAIN_PROMPT = PromptTemplate.from_template("""根据以下已知信息回答问题：
{context}
问题：{question}
请用原文回答，如果无法回答请说不知道。""")
```

3. Sometimes, the LLM adds the prefix '答案' ('Answer') to its responses, so post-processing is applied to remove that prefix and any content before it.

```
str = "答案" # "答案如下","答案是","回答"
def clean_answer(str,id,text):
    if str in text:
        if text.split(str)[0]!="":
        return text[text.find(str)+3:].strip() if text[text.find(str)+2:].strip().startswith((":", "：", ".", "。", ")", "）")) else text[text.find(str)+2:].strip()
    return text
```

#### Results

##### ROUGE

| Original Version | ROUGE-1 (Recall) | ROUGE-1 (Precision) | ROUGE-1 (F1) | ROUGE-2 (R) | ROUGE-2 (P) | ROUGE-2 (F) | ROUGE-L (R) | ROUGE-L (P) | ROUGE-L (F) |
|------------------|------------------|---------------------|--------------|-------------|-------------|-------------|-------------|-------------|-------------|
| deepseek-chat    | 0.8761           | 0.8877              | 0.8689       | 0.8630      | 0.8564      | 0.8548      | 0.8699      | 0.8821      | 0.8624      |
| glm-4-plus       | 0.8519           | 0.8747              | 0.8468       | 0.8424      | 0.8426      | 0.8350      | 0.8469      | 0.8677      | 0.8394      |
| qwen-max         | 0.8329           | 0.8948              | 0.8354       | 0.8271      | 0.8347      | 0.8270      | 0.8308      | 0.8928      | 0.8318      |

| Version (with `clean_answer()`) | ROUGE-1 (Recall) | ROUGE-1 (Precision) | ROUGE-1 (F1) | ROUGE-2 (R) | ROUGE-2 (P) | ROUGE-2 (F) | ROUGE-L (R) | ROUGE-L (P) | ROUGE-L (F) |
|---------------------------------|------------------|---------------------|--------------|-------------|-------------|-------------|-------------|-------------|-------------|
| deepseek-chat    | 0.8755           | 0.9028              | 0.8767       | 0.8629      | 0.8703      | 0.8633      | 0.8694      | 0.8956      | 0.8693      |
| glm-4-plus       | 0.8518           | 0.8753              | 0.8470       | 0.8417      | 0.8432      | 0.8353      | 0.8468      | 0.8683      | 0.8396      |
| qwen-max         | 0.8324           | 0.8968              | 0.8366       | 0.8260      | 0.8367      | 0.8284      | 0.8302      | 0.8947      | 0.8331      |

##### BERTSCORE

| Model      | Precision Score | Recall Score | F1 Score  |
|------------|-----------------|--------------|-----------|
| deepseek   | 0.9323          | 0.9262       | 0.9276    |
| glm-4-plus | 0.9253          | 0.9097       | 0.9153    |
| qwen-max   | 0.9203          | 0.8994       | 0.9080    |

If the model's answer contains "不知道" ("I don't know"), it is considered incorrect. Below is the error rate:

| Model         | error    |
|---------------|----------|
| deepseek-chat | 7.5314%  |
| glm-4-plus    | 11.1576% |
| qwen-max      | 14.9233% |

## Running Examples

![](https://github.com/WillongWang/Local-KB-QA-with-RAG/blob/main/1.png)

![](https://github.com/WillongWang/Local-KB-QA-with-RAG/blob/main/2.png)

### Bugs

Currently, only document uploads from a folder are supported, and all files in the folder will be fully uploaded, because:

```
# The text_spliter is ineffective for TextLoader, and the documents have not been split.
loader = TextLoader(directory,encoding='utf-8') 
documents = loader.load()
text_spliter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
split_docs = text_spliter.split_documents(documents)

# Must use 
loader = DirectoryLoader(directory)
```