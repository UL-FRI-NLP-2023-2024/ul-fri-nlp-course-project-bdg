import re

import numpy as np
import pandas as pd
import os
import sys
import json

path_to_data = os.path.join('../../datasets/dataset_finance/')
path_to_save_corpus = os.path.join('../../corpora/corpus_finance/')
path_to_save_our_model = os.path.join('../../corpora_model_fine-tune/corpus_model_fine-tune_finance')

all_files_count = len(os.listdir(path_to_data))
not_dialogue_count = 0

list_corpus = []
list_our_model = []

corpus_index = 0
our_model_index = 0


index_files = 1
for filename in os.listdir(path_to_data):
    #if filename == 't39073.json':
    #    print('exception')
    #filename = '2676836.json'

    load_path = ""
    load_path = os.path.join(path_to_data, filename)
    print(f'Preprocessing {load_path},  {index_files}/{all_files_count}')
    index_files += 1
    data = pd.read_json(load_path)

    # check if there is only 1 user
    if (data.shape[0] < 2):
        print(f'File {filename} has only 1 user - not a dialogue')
        not_dialogue_count += 1
        continue

    # rename 'author' column to 'user'
    data = data.rename(columns={'author': 'user'})
    # rename 'content' column to 'message'
    data = data.rename(columns={'content': 'message'})


    # change first user to "user" and do that for the whole dataset
    user1 = str(data['user'][0])
    data['user'] = data['user'].replace(user1, 'user')

    # remove all html tags
    # remove \t
    data['message'] = data['message'].str.replace('\t', ' ')
    # remove \n
    data['message'] = data['message'].str.replace('\n', ' ')
    # strip extra spaces
    data['message'] = data['message'].str.strip()
    data['message'] = data['message'].str.replace('  ', ' ')

    # change all other users to assistant_n
    n = 0
    assistants_dict = {}
    assistants_dict[user1] = 'user'
    for user in data['user']:
        #user = user.lower()
        # not the first user
        if (user == 'user'):
            continue

        # already named assistant
        if (user in assistants_dict):
            continue

        # new assistant
        assistants_dict[user] = 'assistant_' + str(n)
        data['user'] = data['user'].replace(user, 'assistant_' + str(n))
        #data['user'] = data['user'].str.lower()
        n += 1

    # replace names in the message
    for user in assistants_dict:
        asist_user = "@"+assistants_dict[user]
        #msg = data_lowercase_copy['message']
        #msg = msg.str.replace(str(user), asist_user)
        #data_lowercase_copy['message'] = msg
        #data['message'] = msg
        data['message'] = data['message'].str.replace(str(user), asist_user)

    # remove double @
    data['message'] = data['message'].str.replace('@@', '@')

    # remove empty spaces
    data['message'] = data['message'].str.strip()

    # remove empty msg rows
    data = data[data['message'] != '']
    # reset index
    data = data.reset_index(drop=True)

    # skip conversation with only one message
    if (data.shape[0] < 2):
        not_dialogue_count += 1
        continue



    # save the preprocessed data - corpus
    source = "finance"
    category = "unknown"
    prompt = ""
    prompt_user = "user"
    answers = []

    # define list of words that are considered as question
    question_words = []
    with open('../regex_lists/question_words.txt', 'r', encoding='utf-8') as file:
        question_words = [line.strip() for line in file]

    # prepare regex patterns for citations
    citation_pattern_start_1 = r'.*Citat: Uporabnik .*? pravi:'
    citation_pattern_start_2 = r'.*Uporabnik .*? je napisal:'
    citation_pattern_start_1_exact = r'Citat: Uporabnik .*? pravi:'
    citation_pattern_start_2_exact = r'Uporabnik .*? je napisal:'
    citation_pattern_end = r'Klikni za razširitev'
    citation_pattern_whole_1 = r'{}.*{}'.format(citation_pattern_start_1, citation_pattern_end)
    citation_pattern_whole_2 = r'{}.*{}'.format(citation_pattern_start_2, citation_pattern_end)

    for index, row in data.iterrows():

        msg = row['message']
        msg = msg.strip()

        citation_match_1 = re.match(citation_pattern_whole_1, msg)
        citation_match_2 = re.match(citation_pattern_whole_2, msg)

        # first prompt
        if (index == 0):
            # add contect to first prompt message
            prompt = row['ctx'] + ". " + row['message']

        # special case - citation
        elif (citation_match_1 or citation_match_2):

            while(msg != ""):
                # only 1 citation
                # get the citation between the start and stop pattern
                citation_prompt = ""
                citation_answer = ""

                # find the prompt
                citation_start = re.search(citation_pattern_start_1_exact, msg)
                if (citation_start == None):
                    citation_start = re.search(citation_pattern_start_2_exact, msg)
                citation_end = re.search(citation_pattern_end, msg)

                citation_prompt = msg[citation_start.regs[0][1]:citation_end.regs[0][0]]
                # find the answer
                citation_answer = msg[citation_end.regs[0][1]:]

                # does answer contain citation?
                citation_match_1 = re.match(citation_pattern_whole_1, citation_answer)
                citation_match_2 = re.match(citation_pattern_whole_2, citation_answer)
                if (citation_match_1 or citation_match_2):
                    # find new start
                    citation_start = re.search(citation_pattern_start_1_exact, citation_answer)
                    if (citation_start == None):
                        citation_start = re.search(citation_pattern_start_2_exact, citation_answer)

                    msg = citation_answer[citation_start.regs[0][0]:]
                    citation_answer = citation_answer[:citation_start.regs[0][0]]
                else:
                    msg = ""


                # save the citation
                list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': citation_prompt.strip(), 'answers': [{'role': "assistant", 'message': citation_answer.strip(), "answer_rating": row["role_rating"], "answer_hate": 0}]})
                corpus_index += 1


        # special case - mention assistant
        elif ("@assistant" in row['message']):
            exact_assistant = re.search(r'@assistant_(\d+)', row['message']).group(1)
            exact_assistant = "assistant_" + str(exact_assistant)

            # check if its talking about himself
            if (exact_assistant == row['user']):
                # remove number
                row['message'] = row['message'].replace("@" + exact_assistant, "")
                # if user
                if (row['user'] == prompt_user):
                    # save current one
                    if (len(answers) > 0):
                        list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': prompt.strip(), 'answers': answers})
                        corpus_index += 1
                    # start new
                    prompt = row['message']
                else:
                    answers.append({'role': "assistant", 'message': row['message'].strip(), "answer_rating": row['role_rating'], "answer_hate": 0})
            else:
                mention_prompt = ""
                mention_answer = row['message']
                # change assistant_n to "user"
                mention_answer = mention_answer.replace("@" + exact_assistant, "@user")

                # find previous message from this assistant
                for i in range(index-1, -1, -1):
                    if (data['user'][i] == exact_assistant):
                        mention_prompt = data['message'][i]
                        break

                # if not found
                if (mention_prompt == ""):
                    continue

                # check if prompt has mention and change is to @assistant
                if ("@assistant" in mention_answer):
                    exact_assistant = re.search(r'@assistant_(\d+)', mention_answer).group(1)
                    exact_assistant = "assistant_" + str(exact_assistant)
                    special_answer = mention_answer.replace("@" + exact_assistant, "@user")
                if ("@user" in mention_prompt):
                    special_prompt = mention_prompt.replace("@user", "@assistant")

                list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': mention_prompt.strip(), 'answers': [{'role': "assistant", 'message': mention_answer.strip(), "answer_rating": row["role_rating"], "answer_hate": 0}]})
                corpus_index += 1

        # special case - mention user
        elif ("@user" in row['message']):
            exact_user = "user"
            # check if its talking about himself
            if (exact_user == row['user']):
                # remove number
                row['message'] = row['message'].replace("@" + exact_user, "")
                # if user
                if (row['user'] == prompt_user):
                    # save current one
                    if (len(answers) > 0):
                        list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': prompt.strip(), 'answers': answers})
                        corpus_index += 1
                    # start new
                    prompt = row['message']
                else:
                    answers.append({'role': "assistant", 'message': row['message'].strip(), "answer_rating": row["role_rating"], "answer_hate": 0})
            else:
                mention_prompt = ""
                mention_answer = row['message']

                # find previous message from this user
                for i in range(index-1, -1, -1):
                    if (data['user'][i] == exact_user):
                        mention_prompt = data['message'][i]
                        break

                # if not found
                if (mention_prompt == ""):
                    continue

                # check if prompt has mention and change is to @assistant
                if ("@assistant" in mention_answer):
                    exact_user = "user"
                    exact_assistant = re.search(r'@assistant_(\d+)', mention_answer).group(1)
                    mention_answer = mention_answer.replace("@" + exact_assistant, "@user")
                if ("@user" in mention_prompt):
                    mention_prompt = mention_prompt.replace("@user", "@assistant")

                list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': mention_prompt.strip(), 'answers': [{'role': "assistant", 'message': mention_answer.strip(), "answer_rating": row["role_rating"], "answer_hate": 0}]})
                corpus_index += 1

        # new prompt
        elif (row['user'] == prompt_user):

            # check if this is a question: if yes this is new prompt else this is answe to previous message
            # use question_words.txt

            is_question = False
            for word in question_words:
                if word.lower() in row['message'].lower():
                    is_question = True
                    break

            if (is_question):
                # new prompt
                # save current one
                if (len(answers) > 0):
                    list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': prompt.strip(), 'answers': answers})
                    corpus_index += 1

                # start new
                prompt = row['message']
                answers = []
            else:
                # answer
                new_answer = [{'role': "assistant", 'message': row['message'].strip(), "answer_rating": row["role_rating"], "answer_hate": 0}]
                # prompt is previous
                new_prompt = data['message'][index-1]
                list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': new_prompt.strip(), 'answers': new_answer})


        # answer
        else:
            answers.append({'role': "assistant", 'message': row['message'].strip(), "answer_rating": row["role_rating"], "answer_hate": 0})

    # save the last one

    # if answers are empty - reply to previous message
    if (len(answers) == 0):
        answers.append({'role': "assistant", 'message': prompt.strip(), "answer_rating": data.iloc[-1]["role_rating"], "answer_hate": 0})
        # prompt is one before last
        prompt = data['message'][data.shape[0]-2]
    list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': prompt.strip(), 'answers': answers})
    corpus_index += 1


# check all prompts in corpus if they have @assistant_n and remove the _n
for entry in list_corpus:
    entry['prompt'] = re.sub(r'@assistant_\d+', '@assistant', entry['prompt'])

# check for hate words - increase for every word in answer
# use blocked_words.txt
hate_words = []
with open('../regex_lists/blocked_words.txt', 'r', encoding='utf-8') as file:
    hate_words = [line.strip() for line in file]

for entry in list_corpus:
    for answer in entry['answers']:
        words = answer['message'].split()
        hate_word_count = 0
        for word in words:
            for hate_word in hate_words:
                if word.lower().startswith(hate_word):
                    hate_word_count += 1

        answer['answer_hate'] = hate_word_count


# can be reworked and moved to another script for fine-tunning
# save the preprocessed data, each conversation into individual file - our model finetuning
prompt_user = "user"
prompt = ""
answer_user = "assistant"
preprocessed_data = []
i = 0
for entry in list_corpus:
    print(f'Saving {i}/{len(list_corpus)}')
    # save the prompt
    prompt = entry['prompt']
    answers = entry['answers']
    for answer in answers:
        preprocessed_data = []
        preprocessed_data.append({'role': "user", 'content': prompt})
        preprocessed_data.append({'role': "assistant", 'content': answer['message']})
        list_our_model.append(preprocessed_data)



# save the lists
save_path = os.path.join(path_to_save_corpus, "corpus_finance.json")
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(list_corpus, f, indent=4, ensure_ascii=False)

save_path = os.path.join(path_to_save_our_model, "model_finetune_fincance.json")
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(list_our_model, f, indent=4, ensure_ascii=False)



num_corpus_files = len(os.listdir(path_to_save_corpus))
num_our_model_files = len(os.listdir(path_to_save_our_model))
print(f'Preprocessed {all_files_count} files out of {all_files_count}')
print(f'Files with no dialogue: {not_dialogue_count}')
print(f'Preprocessed {len(list_corpus)} files into corpus')
print(f'Preprocessed {len(list_our_model)} files into our list for our model finetuning')
