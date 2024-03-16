from transformers import pipeline
import pandas
from datetime import datetime

translator = pipeline("translation", model="duongy18418/Multilingual_Model")

ecommerce_dataset = pandas.read_csv("./Code/Datasets/Test/Amazon_Ecommerce_Data_2020.csv", usecols=["Category"], nrows=100)
ecommerce_dataset = ecommerce_dataset.dropna()
en_list = ecommerce_dataset['Category'].tolist()
zh_list = []

for i in range(len(en_list)):
    zh_list.append(translator(en_list[i], src_lang='en', tgt_lang='zh'))

ecommerce_dataset = pandas.DataFrame(list(zip(en_list, zh_list)), columns=['Category-en', 'Category-zh'])
currentDateTime = datetime.now().strftime("%m-%d-%Y %H-%M-%S %p")
ecommerce_dataset.to_csv(f'./Code/Results/result {currentDateTime}.csv', index=False)
