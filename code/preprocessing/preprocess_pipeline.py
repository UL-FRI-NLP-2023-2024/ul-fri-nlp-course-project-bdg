import numpy as np
import pandas as pd
import os
import sys
import json

path_to_data = os.path.join('./data')
path_to_save = os.path.join('./preprocessed_data')

all_files_count = len(os.listdir(path_to_data))
not_dialogue_count = 0

for filename in os.listdir(path_to_data):
    #if filename == 't39073.json':
    #    print('exception')

    load_path = ""
    load_path = os.path.join(path_to_data, filename)
    print(f'Preprocessing {load_path}')
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

    # change all other users to assistant_n
    n = 0
    assistants_dict = {}
    assistants_dict[user1] = 'user'
    for user in data['user']:
        # not the first user
        if (user == 'user'):
            continue

        # already named assistant
        if (user in assistants_dict):
            continue

        # new assistant
        assistants_dict[user] = 'assistant_' + str(n)
        data['user'] = data['user'].replace(user, 'assistant_' + str(n))
        n += 1


    # replace names in the message
    for user in assistants_dict:
        asist_user = "@"+assistants_dict[user]
        data['message'] = data['message'].str.replace(str(user), asist_user)


    # remove double @
    data['message'] = data['message'].str.replace('@@', '@')


    # remove empty spaces
    data['message'] = data['message'].str.strip()


    # save the preprocessed data
    preprocessed_data = []
    for index, row in data.iterrows():
        user = row['user']
        message = row['message']
        dict = {'role': user, 'message': message}
        preprocessed_data.append(dict)


    save_path = ""
    save_path = os.path.join(path_to_save, filename)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(preprocessed_data, f, indent=4, ensure_ascii=False)

print(f'Preprocessed {all_files_count - not_dialogue_count} dialogues out of {all_files_count} files. {not_dialogue_count} files were not dialogues.')