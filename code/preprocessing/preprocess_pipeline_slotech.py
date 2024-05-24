import re

import numpy as np
import pandas as pd
import os
import sys
import json

path_to_data = os.path.join('../../datasets/dataset_slotech/')
path_to_save_corpus = os.path.join('../../corpora/corpus_slotech/')
path_to_save_our_model = os.path.join('../../corpora_model_fine-tune/corpus_model_fine-tune_slotech')

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
    #filename = 't150345.json'

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

    # replace \" with "
    #data['message'] = data['message'].str.replace('\"', '"')

    # change first user to "user" and do that for the whole dataset
    user1 = str(data['user'][0])
    data['user'] = data['user'].replace(user1, 'user')

    #data_lowercase_copy = data.copy()
    #data_lowercase_copy = data_lowercase_copy.apply(lambda x: x.astype(str).str.lower())

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
    #preprocessed_data = []
    source = "slo-tech"
    category = "unknown"
    prompt = ""
    prompt_user = "user"
    answers = []

    question_words = []
    with open('../regex_lists/question_words.txt', 'r', encoding='utf-8') as file:
        question_words = [line.strip() for line in file]

    #print(question_words)
    #print(data['message'][0])

    citation_pattern_assistant = r'.*@assistant_\d+ je .* \d{4} ob \d{2}:\d{2} izjavil:'
    citation_pattern_assistant_exact = r'@assistant_\d+ je .* \d{4} ob \d{2}:\d{2} izjavil:'
    nested_citation_pattern_assistant = r'.*@assistant_\d+ je .* \d{4} ob \d{2}:\d{2} izjavil:@assistant_\d+ je .* \d{4} ob \d{2}:\d{2} izjavil:'

    citation_pattern_user = r'.*@user je .* \d{4} ob \d{2}:\d{2} izjavil:'
    citation_pattern_user_exact = r'@user je .* \d{4} ob \d{2}:\d{2} izjavil:'
    nested_citation_pattern_user = r'.*@user je .* \d{4} ob \d{2}:\d{2} izjavil:@user je .* \d{4} ob \d{2}:\d{2} izjavil:'

    for index, row in data.iterrows():

        # trim extra spaces
        #row['message'] = row['message'].strip()

        #if "Fak pa od kje ste potegnili to temo? Drugače se pa strinjam z" in row['message']:
            #print('exception')

        msg = row['message']
        citation_match_assistant = re.match(citation_pattern_assistant, msg)
        citation_match_user = re.match(citation_pattern_user, msg)

        # first prompt
        if (index == 0):
            prompt = row['message']

        # special case - citation assistant
        elif (citation_match_assistant):

            # if nested citation skip it - already handled in other conversation
            match_nested = re.match(nested_citation_pattern_assistant, msg)
            if (match_nested):
                continue

            next_match = citation_match_assistant
            while (next_match):
                citation_prompt = ""
                citation_answer = ""
                cited_assistant = re.search(r'@assistant_(\d+)', next_match.group(0)).group(1)
                all_msgs_of_cited_assistant = data[data['user'] == 'assistant_' + cited_assistant]
                for msg_of_cited_assistant in all_msgs_of_cited_assistant['message']:
                    msg_to_match = msg_of_cited_assistant
                    # find substring msg_to_match in msg
                    if (msg_to_match in msg):
                        # remove msg_to_match from msg and everything before that
                        msg = msg[msg.find(msg_to_match) + len(msg_to_match):]
                        citation_prompt = msg_to_match
                        break

                # no match found
                if (citation_prompt == ""):
                    break

                next_match = re.match(citation_pattern_assistant, msg)
                if (next_match):
                    citation_answer = msg[next_match.regs[0][0]:next_match.regs[0][1]]
                    # remove the pattern from citation answer
                    citation_answer = re.sub(citation_pattern_assistant_exact, '', citation_answer)

                    # remove answer from msg
                    msg = msg.replace(citation_answer, "")
                else:
                    citation_answer = msg

                # save the citation
                list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': citation_prompt.strip(), 'answers': [{'role': "assistant", 'message': citation_answer.strip(), "answer_rating": 0, "answer_hate": 0}]})
                corpus_index += 1


        # special case - citation user
        elif (citation_match_user):

                # if nested citation skip it - already handled in other conversation
                match_nested = re.match(nested_citation_pattern_user, msg)
                if (match_nested):
                    continue

                next_match = citation_match_user
                while (next_match):
                    citation_prompt = ""
                    citation_answer = ""
                    cited_user = re.search(r'@user', next_match.group(0)).group(0)
                    all_msgs_of_cited_user = data[data['user'] == 'user']
                    for msg_of_cited_user in all_msgs_of_cited_user['message']:
                        msg_to_match = msg_of_cited_user
                        # find substring msg_to_match in msg
                        if (msg_to_match in msg):
                            # remove msg_to_match from msg and everything before that
                            msg = msg[msg.find(msg_to_match) + len(msg_to_match):]
                            citation_prompt = msg_to_match
                            break

                    # no match found
                    if (citation_prompt == ""):
                        break

                    next_match = re.match(citation_pattern_user, msg)
                    if (next_match):
                        citation_answer = msg[next_match.regs[0][0]:next_match.regs[0][1]]
                        # remove the pattern from citation answer
                        citation_answer = re.sub(citation_pattern_user_exact, '', citation_answer)

                        # remove answer from msg
                        msg = msg.replace(citation_answer, "")
                    else:
                        citation_answer = msg

                    # save the citation
                    list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': citation_prompt.strip(), 'answers': [{'role': "assistant", 'message': citation_answer.strip(), "answer_rating": 0, "answer_hate": 0}]})
                    corpus_index += 1


        # special case - mention assistant
        elif ("@assistant" in row['message']):
            exact_assistant = re.search(r'@assistant_(\d+)', row['message']).group(1)
            exact_assistant = "assistant_" + str(exact_assistant)

            # check if its talking about himself
            if (exact_assistant == row['user']):
                #print('exception - talking about himself on index:', index, 'user:', row['user'], 'message:', row['message'])
                # remove number
                row['message'] = row['message'].replace("@" + exact_assistant, "")
                # if user
                if (row['user'] == prompt_user):
                    # save current one
                    if (len(answers) > 0):
                        list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': prompt.strip(), 'answers': answers})
                        #save_path = os.path.join(path_to_save_corpus, filename + "_" + str(corpus_index))
                        #with open(save_path, 'w', encoding='utf-8') as f:
                        #    json.dump(dict, f, indent=4, ensure_ascii=False)
                        corpus_index += 1
                        #list_corpus.append(dict)
                    # start new
                    prompt = row['message']
                else:
                    answers.append({'role': "assistant", 'message': row['message'].strip(), "answer_rating": 0, "answer_hate": 0})
            else:
                special_prompt = ""
                special_answer = row['message']
                # change assistant_n to "user"
                special_answer = special_answer.replace("@" + exact_assistant, "@user")

                # find previous message from this assistant
                for i in range(index-1, -1, -1):
                    if (data['user'][i] == exact_assistant):
                        special_prompt = data['message'][i]
                        break

                # if not found
                if (special_prompt == ""):
                    continue

                # check if prompt has mention and change is to @assistant
                if ("@assistant" in special_answer):
                    exact_assistant = re.search(r'@assistant_(\d+)', special_answer).group(1)
                    exact_assistant = "assistant_" + str(exact_assistant)
                    special_answer = special_answer.replace("@" + exact_assistant, "@user")
                if ("@user" in special_prompt):
                    special_prompt = special_prompt.replace("@user", "@assistant")

                list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': special_prompt.strip(), 'answers': [{'role': "assistant", 'message': special_answer.strip(), "answer_rating": 0, "answer_hate": 0}]})
                #save_path = os.path.join(path_to_save_corpus, filename + "_" + str(corpus_index))
                #with open(save_path, 'w', encoding='utf-8') as f:
                #    json.dump(dict, f, indent=4, ensure_ascii=False)
                corpus_index += 1
                #list_corpus.append(dict)

        # special case - mention user
        elif ("@user" in row['message']):
            exact_user = "user"
            # check if its talking about himself
            if (exact_user == row['user']):
                #print('exception - talking about himself on index:', index, 'user:', row['user'], 'message:', row['message'])
                # remove number
                row['message'] = row['message'].replace("@" + exact_user, "")
                # if user
                if (row['user'] == prompt_user):
                    # save current one
                    if (len(answers) > 0):
                        list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': prompt.strip(), 'answers': answers})
                        #save_path = os.path.join(path_to_save_corpus, filename + "_" + str(corpus_index))
                        #with open(save_path, 'w', encoding='utf-8') as f:
                        #    json.dump(dict, f, indent=4, ensure_ascii=False)
                        corpus_index += 1
                        #list_corpus.append(dict)
                    # start new
                    prompt = row['message']
                else:
                    answers.append({'role': "assistant", 'message': row['message'].strip(), "answer_rating": 0, "answer_hate": 0})
            else:
                special_prompt = ""
                special_answer = row['message']

                # find previous message from this user
                for i in range(index-1, -1, -1):
                    if (data['user'][i] == exact_user):
                        special_prompt = data['message'][i]
                        break

                # if not found
                if (special_prompt == ""):
                    continue

                # check if prompt has mention and change is to @assistant
                if ("@assistant" in special_answer):
                    exact_user = "user"
                    exact_assistant = re.search(r'@assistant_(\d+)', special_answer).group(1)
                    special_answer = special_answer.replace("@" + exact_assistant, "@user")
                if ("@user" in special_prompt):
                    special_prompt = special_prompt.replace("@user", "@assistant")

                list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': special_prompt.strip(), 'answers': [{'role': "assistant", 'message': special_answer.strip(), "answer_rating": 0, "answer_hate": 0}]})


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
                    #save_path = os.path.join(path_to_save_corpus, filename + "_" + str(corpus_index))
                    #with open(save_path, 'w', encoding='utf-8') as f:
                    #    json.dump(dict, f, indent=4, ensure_ascii=False)
                    corpus_index += 1
                    #list_corpus.append(dict)

                # start new
                prompt = row['message']
                answers = []
            else:
                # answer
                new_answer = [{'role': "assistant", 'message': row['message'].strip(), "answer_rating": 0, "answer_hate": 0}]
                # prompt is previous
                new_prompt = data['message'][index-1]
                list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': new_prompt.strip(), 'answers': new_answer})


        # answer
        else:
            answers.append({'role': "assistant", 'message': row['message'].strip(), "answer_rating": 0, "answer_hate": 0})

    # save the last one

    # if answers are empty - reply to previous message
    if (len(answers) == 0):
        answers.append({'role': "assistant", 'message': prompt.strip(), "answer_rating": 0, "answer_hate": 0})
        # prompt is one before last
        prompt = data['message'][data.shape[0]-2]
    list_corpus.append({'index': corpus_index, 'source': source, "role": "user", 'prompt': prompt.strip(), 'answers': answers})
    #save_path = os.path.join(path_to_save_corpus, filename + "_" + str(corpus_index))
    #with open(save_path, 'w', encoding='utf-8') as f:
    #    json.dump(dict, f, indent=4, ensure_ascii=False)
    corpus_index += 1
    #list_corpus.append(dict)




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

        # also give them all rating of 1, since no upvote metric is present
        answer['answer_rating'] = 1


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
save_path = os.path.join(path_to_save_corpus, "corpus_slotech.json")
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(list_corpus, f, indent=4, ensure_ascii=False)

save_path = os.path.join(path_to_save_our_model, "model_finetune_slotech.json")
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(list_our_model, f, indent=4, ensure_ascii=False)



num_corpus_files = len(os.listdir(path_to_save_corpus))
num_our_model_files = len(os.listdir(path_to_save_our_model))
print(f'Preprocessed {all_files_count} files out of {all_files_count}')
print(f'Files with no dialogue: {not_dialogue_count}')
print(f'Preprocessed {len(list_corpus)} files into corpus')
print(f'Preprocessed {len(list_our_model)} files into our list for our model finetuning')
