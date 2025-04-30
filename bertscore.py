import pandas as pd
from bert_score import score

# 读取原始数据
df = pd.read_csv("comparison_results.csv")
original_answers = df['原答案'].tolist()
generated_answers = df['生成答案'].tolist()

P, R, F1 = score(generated_answers, original_answers, lang='zh', verbose=True) # "zh": "bert-base-chinese"

print(f"System level precision score: {P.mean()}")
print(f"System level recall score: {R.mean()}")
print(f"System level F1 score: {F1.mean()}")

'''
deepseek:
System level precision score: 0.9322527050971985
System level recall score: 0.9261577725410461
System level F1 score: 0.9275500178337097
glm-4-plus:
System level precision score: 0.9253019094467163
System level recall score: 0.9096961617469788
System level F1 score: 0.9152743220329285
qwen-max:
System level precision score: 0.9203027486801147
System level recall score: 0.8994031548500061
System level F1 score: 0.9080446362495422
'''

