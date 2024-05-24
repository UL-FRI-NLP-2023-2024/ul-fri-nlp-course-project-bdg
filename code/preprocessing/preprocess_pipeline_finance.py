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
    #filename = '111450_no_duplicates.json'

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


    # remove invisible characters
    data['message'] = data['message'].str.replace(u'\u00A0', ' ')
    # strip extra spaces
    data['message'] = data['message'].str.strip()



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
    citation_pattern_start_1 = r'.* wrote:'
    citation_pattern_start_2 = r'.* je napisal(a):'
    citation_pattern_start_3 = r'^\[\b(\w+\s){0,2}\w+\b]'

    citation_pattern_start_1_exact = r'^\b(\w+\s){0,2}\w+\b wrote:'
    citation_pattern_start_2_exact = r'^\b(\w+\s){0,2}\w+\b je napisal\(a\):'
    citation_pattern_start_3_exact = r'^@\b(\w+\s){0,2}\w+\b wrote:'
    citation_pattern_start_4_exact = r'^@\b(\w+\s){0,2}\w+\b je napisal\(a\):'

    citation_pattern_start_5_exact = r'^\[\b(\w+\s){0,2}\w+\b]'
    citation_pattern_start_6_exact = r'^\[@\b(\w+\s){0,2}\w+\b]'
    #citation_pattern_start_2_exact = r'[.*?]'
    #citation_pattern_end = r'Klikni za razširitev'
    #citation_pattern_whole_1 = r'{}.*{}'.format(citation_pattern_start_1, citation_pattern_end)
    #citation_pattern_whole_2 = r'{}.*{}'.format(citation_pattern_start_2, citation_pattern_end)
    edit_pattern1 = r'^Zadnja sprememba: @assistant_\d+ .*? \d{2}:\d{2}'
    edit_pattern2 = r'^Zadnja sprememba: @user .*? \d{2}:\d{2}'

    for index, row in data.iterrows():

        msg = row['message']
        msg = msg.strip()

        # match edit pattern and remove it
        edit_match1 = re.match(edit_pattern1, msg)
        if (edit_match1):
            msg = msg.replace(edit_match1.group(0), "")
            msg = msg.strip()
        else :
            edit_match2 = re.match(edit_pattern2, msg)
            if (edit_match2):
                msg = msg.replace(edit_match2.group(0), "")
                msg = msg.strip()

        citation_match_1 = re.match(citation_pattern_start_1_exact, msg)
        citation_match_2 = re.match(citation_pattern_start_2_exact, msg)
        citation_match_3 = re.match(citation_pattern_start_3_exact, msg)
        citation_match_4 = re.match(citation_pattern_start_4_exact, msg)
        citation_match_5 = re.match(citation_pattern_start_5_exact, msg)
        citation_match_6 = re.match(citation_pattern_start_6_exact, msg)

        # first prompt
        if (index == 0):
            # add contect to first prompt message
            prompt = row['ctx'] + ". " + row['message']

        # special case - citation
        elif (citation_match_1 or citation_match_2 or citation_match_3 or citation_match_4 or citation_match_5 or citation_match_6):

            # extra handle for citation match 3
            #if (citation_match_3):
            #    print("exception")

            #while(msg != ""):
            # only 1 citation
            # get the citation between the start and stop pattern
            citation_prompt = ""
            citation_answer = ""

            # find the prompt
            # find correct start pattern
            citation_start = None
            if (citation_match_1):
                citation_start = re.match(citation_pattern_start_1_exact, msg)
            elif (citation_match_2):
                citation_start = re.match(citation_pattern_start_2_exact, msg)
            elif (citation_match_3):
                citation_start = re.match(citation_pattern_start_3_exact, msg)
            elif (citation_match_4):
                citation_start = re.match(citation_pattern_start_4_exact, msg)
            elif (citation_match_5):
                citation_start = re.match(citation_pattern_start_5_exact, msg)
            elif (citation_match_6):
                citation_start = re.match(citation_pattern_start_6_exact, msg)

            if (citation_start == None):
                print("exception")
            else:
                # remove starging pattern
                #citation_prompt = msg.replace(citation_start.group(0), "").strip()

                # search for citation source in previous messages
                for i in range(index-1, -1, -1):
                    prev_msg = data['message'][i]
                    if ((msg.replace(citation_start.group(0), "").strip()).startswith(prev_msg)):
                        citation_prompt = data['message'][i]
                        break

                if (citation_prompt == ""):
                    # no quite found - use as prompt/anwser
                    if (row['user'] == prompt_user):
                        # save current one
                        if (len(answers) > 0):
                            list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': prompt.strip(), 'answers': answers})
                            corpus_index += 1
                        # start new
                        prompt = msg
                        continue
                    else:
                        # save as answer
                        answers.append({'role': "assistant", 'message': msg.strip(), "answer_rating": int(row["upvotes"]), "answer_hate": 0})
                        continue
                else:
                    # citation is msg - stargin_pattern - citation_prompt
                    citation_answer = msg.replace(citation_start.group(0), "").strip()
                    citation_answer = citation_answer.replace(citation_prompt, "").strip()
                    citation_answer = citation_answer.strip()

                    if (citation_answer == ""):
                        print("exception")
                        continue

                # into dataframe save only the answer without citation for next iteration
                data.at[index, 'message'] = citation_answer

                # save the citation
                list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': citation_prompt.strip(), 'answers': [{'role': "assistant", 'message': citation_answer.strip(), "answer_rating": int(row["upvotes"]), "answer_hate": 0}]})
                corpus_index += 1


        # special case - mention assistant
        elif ("@assistant" in msg):
            exact_assistant = re.search(r'@assistant_(\d+)', msg).group(1)
            exact_assistant = "assistant_" + str(exact_assistant)

            # check if its talking about himself
            if (exact_assistant == row['user']):
                # remove number
                msg = msg.replace("@" + exact_assistant, "")
                # if user
                if (row['user'] == prompt_user):
                    # save current one
                    if (len(answers) > 0):
                        list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': prompt.strip(), 'answers': answers})
                        corpus_index += 1
                    # start new
                    prompt = msg
                else:
                    answers.append({'role': "assistant", 'message': msg.strip(), "answer_rating": int(row['upvotes']), "answer_hate": 0})
            else:
                mention_prompt = ""
                mention_answer = msg
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
                    mention_answer = mention_answer.replace("@" + exact_assistant, "@user")
                if ("@user" in mention_prompt):
                    mention_prompt = mention_prompt.replace("@user", "@assistant")

                list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': mention_prompt.strip(), 'answers': [{'role': "assistant", 'message': mention_answer.strip(), "answer_rating": int(row["upvotes"]), "answer_hate": 0}]})
                corpus_index += 1

        # special case - mention user
        elif ("@user" in msg):
            exact_user = "user"
            # check if its talking about himself
            if (exact_user == row['user']):
                # remove number
                msg = msg.replace("@" + exact_user, "")
                # if user
                if (row['user'] == prompt_user):
                    # save current one
                    if (len(answers) > 0):
                        list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': prompt.strip(), 'answers': answers})
                        corpus_index += 1
                    # start new
                    prompt = msg
                else:
                    answers.append({'role': "assistant", 'message': msg.strip(), "answer_rating": int(row["upvotes"]), "answer_hate": 0})
            else:
                mention_prompt = ""
                mention_answer = msg

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

                list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': mention_prompt.strip(), 'answers': [{'role': "assistant", 'message': mention_answer.strip(), "answer_rating": int(row["upvotes"]), "answer_hate": 0}]})
                corpus_index += 1

        # new prompt
        elif (row['user'] == prompt_user):

            # check if this is a question: if yes this is new prompt else this is answe to previous message
            # use question_words.txt

            is_question = False
            for word in question_words:
                if word.lower() in msg.lower():
                    is_question = True
                    break

            if (is_question):
                # new prompt
                # save current one
                if (len(answers) > 0):
                    list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': prompt.strip(), 'answers': answers})
                    corpus_index += 1

                # start new
                prompt = msg
                answers = []
            else:
                # answer
                new_answer = [{'role': "assistant", 'message': msg.strip(), "answer_rating": int(row["upvotes"]), "answer_hate": 0}]
                # prompt is previous
                new_prompt_msg = data['message'][index-1]
                new_prompt_msg = new_prompt_msg.strip()

                # match edit pattern and remove it
                edit_match1 = re.match(edit_pattern1, new_prompt_msg)
                if (edit_match1):
                    new_prompt_msg = new_prompt_msg.replace(edit_match1.group(0), "")
                    new_prompt_msg = new_prompt_msg.strip()
                else:
                    edit_match2 = re.match(edit_pattern2, new_prompt_msg)
                    if (edit_match2):
                        new_prompt_msg = new_prompt_msg.replace(edit_match2.group(0), "")
                        new_prompt_msg = new_prompt_msg.strip()

                new_prompt = new_prompt_msg
                list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': new_prompt.strip(), 'answers': new_answer})


        # answer
        else:
            answers.append({'role': "assistant", 'message': msg.strip(), "answer_rating": int(row["upvotes"]), "answer_hate": 0})

    # save the last one

    # if answers are empty - reply to previous message
    if (len(answers) == 0):
        answers.append({'role': "assistant", 'message': prompt.strip(), "answer_rating": int(data.iloc[-1]["upvotes"]), "answer_hate": 0})
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
    i += 1
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

save_path = os.path.join(path_to_save_our_model, "model_finetune_finance.json")
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(list_our_model, f, indent=4, ensure_ascii=False)



num_corpus_files = len(os.listdir(path_to_save_corpus))
num_our_model_files = len(os.listdir(path_to_save_our_model))
print(f'Preprocessed {all_files_count} files out of {all_files_count}')
print(f'Files with no dialogue: {not_dialogue_count}')
print(f'Preprocessed {len(list_corpus)} files into corpus')
print(f'Preprocessed {len(list_our_model)} files into our list for our model finetuning')
