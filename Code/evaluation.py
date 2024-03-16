from sklearn.model_selection import train_test_split
from deep_translator import GoogleTranslator
from transformers import pipeline
import pandas
import time
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt

baseline_model = GoogleTranslator(source='auto', target='zh-CN')
evaluating_model = translator = pipeline("text2text-generation", model="duongy18418/E-commerce_Translation_Model")

ecommerce_dataset = pandas.read_csv("./Code/Datasets/Test/Amazon_Ecommerce_Data_2020.csv", usecols=["Category"], nrows=100)
ecommerce_dataset = ecommerce_dataset.dropna()

base_en = ecommerce_dataset['Category'].tolist()
base_zh = []

start_time = time.time()
for i in range(len(base_en)):
    base_zh.append(baseline_model.translate(base_en[i]))
end_time = time.time()
base_model_time = end_time - start_time

base_dataset = pandas.DataFrame(list(zip(base_en, base_zh)), columns=['Category-en', 'Category-zh'])
base_dataset.to_csv(f'./Code/Results/base_result.csv', index=False)

eval_en = ecommerce_dataset['Category'].tolist()
eval_zh = []

start_time = time.time()
for i in range(len(eval_en)):
    eval_zh.append(str(evaluating_model(eval_en[i])))
end_time = time.time()
eval_model_time = end_time - start_time

eval_dataset = pandas.DataFrame(list(zip(eval_en, eval_zh)), columns=['Category-en', 'Category-zh'])
eval_dataset.to_csv(f'./Code/Results/eval_result.csv', index=False)

base_dataset['BLEU Score'] = base_dataset.apply(lambda row: sentence_bleu(row['Category-en'], row['Category-zh']), axis=1)
eval_dataset['BLUE Score'] = eval_dataset.apply(lambda row: sentence_bleu(row['Category-en'], row['Category-zh']), axis=1)

print("Baseline model runtime:", base_model_time, "seconds")
print("E-Commerce model runtime:", eval_model_time, "seconds")

ax = plt.subplot()
ax.plot(base_dataset['BLEU Score'], label='Baseline Model')
ax.plot(eval_dataset['BLUE Score'], label='E-Commerce Model')
ax.set_xlabel('Index')
ax.set_ylabel('BLEU Score')
ax.set_title("BLEU Score Comparision Between Two Models")
ax.legend()
plt.show()