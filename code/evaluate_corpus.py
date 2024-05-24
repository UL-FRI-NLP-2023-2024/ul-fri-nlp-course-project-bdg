import json
import os

slotech_path = '../corpora/corpus_slotech/corpus_slotech.json'
finance_path = '../corpora/corpus_finance/corpus_finance.json'
alter_path = '../corpora/corpus_alter/corpus_alter.json'

combined_corpus_path = '../corpora/corpus_combined/corpus_combined.json'
combined_corpus_list = []

def read_and_append(file_path, data_list):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data_list.extend(data)

# Read and combine data from all JSON files
read_and_append(slotech_path, combined_corpus_list)
read_and_append(finance_path, combined_corpus_list)
read_and_append(alter_path, combined_corpus_list)

# count number of conversations
num_conversations = len(combined_corpus_list)

# change index of each conversation
for i, conversation in enumerate(combined_corpus_list):
    conversation['index'] = i

# get number of tokens in all conversations
num_tokens = 0
for conversation in combined_corpus_list:

    # tokenize prompt in slovene
    prompt = conversation['prompt']
    num_tokens += len(prompt.split())

    # tokenize all answers in slovene
    answers = conversation['answers']
    for answer in answers:
        answer_msg = answer['message']
        num_tokens += len(answer_msg.split())


print(f'Combined all 3 corpora into 1 large corpus')
print(f'Number of conversations: {num_conversations}')
print(f'Number of tokens: {num_tokens}')

# Write the combined data to a new JSON file
with open(combined_corpus_path, 'w', encoding='utf-8') as f:
    json.dump(combined_corpus_list, f, ensure_ascii=False, indent=4)

print(f"Combined data saved to {combined_corpus_path}")
print()

# print 10 random conversations
import random
random.seed(3)
random_conversations = random.sample(combined_corpus_list, 10)
print(f"Printing 10 random conversations:")
for i, conversation in enumerate(random_conversations):
    print(f"Conversation {i+1}:")
    print(f"Prompt: {conversation['prompt']}")
    print(f"Answers:")
    for j, answer in enumerate(conversation['answers']):
        print(f"Answer {j+1}: {answer['message']}")
    print()


# nltk tokenization in slovenian

print("Tokenizing all conversations using NLTK...")
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# count number of tokens in all conversations
num_tokens = 0

i = 0
for conversation in combined_corpus_list:
    i += 1
    print(f"Tokenizing conversation {i}/{num_conversations}")

    # tokenize prompt in slovene
    prompt = conversation['prompt']
    num_tokens += len(word_tokenize(prompt, language='slovene'))

    # tokenize all answers in slovene
    answers = conversation['answers']
    for answer in answers:
        answer_msg = answer['message']
        num_tokens += len(word_tokenize(answer_msg, language='slovene'))

print()
print(f'Number of tokens after NLTK tokenization: {num_tokens}')

