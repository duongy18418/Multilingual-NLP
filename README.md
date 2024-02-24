# COMP-8730 Project: Translation NLP for E-commerce Product Category

## 1. Project Overview
This project aims to improve the current translation algorithm for any online shopping content translation from English to Chinese. The model developed for this project will be a fine-tuned version of the NLLB-200 translation model provided by Meta.

1. NLLB-200 Overview] https://ai.meta.com/research/no-language-left-behind/
2. NLLB-200 model: https://huggingface.co/facebook/nllb-200-distilled-600M

## 2. Prerequisite
1. Install [Python 3.11.8](https://www.python.org/downloads/release/python-3118/)
2. Install Requirements.txt ```pip install -r Requirements.txt```\
   a. If that doesn't work, run ```pip3 install -r Requirements.txt``` instead.
3. Navigate to the Code folder.
4. Run the model_training.py to start finetuning the model
5. Once the finetuning process is done, run main.py to test the model.

## 3. Reference
1. [NLLB-200 Overview](https://ai.meta.com/research/no-language-left-behind/)
2. [NLLB-200 model](https://huggingface.co/facebook/nllb-200-distilled-600M)
3. [Fine-tune a pretrained model](https://huggingface.co/docs/transformers/en/training)
4. [Translation](https://huggingface.co/docs/transformers/tasks/translation)

## 4. Credits
1. [Y Duong](https://www.linkedin.com/in/y-duong-880140195/)
2. MiaoMiao Zhang
