import pickle
import unicodedata
from tqdm import tqdm
import os
import argparse
import json

# from ipdb import set_trace

import emoji

def encode_dataset2file(lm_char_dict, lm_type_dict, input_folder, output_folder):

    lm_char_type_dict = dict()
    for k, v in lm_char_dict.items():
        if k.islower():
            lm_char_type_dict[k] = lm_type_dict['<low>']
        elif k.isdigit():
            lm_char_type_dict[k] = lm_type_dict['<num>']
        else:
            lm_char_type_dict[k] = lm_type_dict['<pun>']

    if os.path.exists('tmp.json'):
        with open('tmp.json', 'r') as fin:
            ds = json.load(fin)
            complicated_list = ds['complicated_list']
            range_ind = ds['range_ind']
    else:
        complicated_list = list()
        range_ind = 0

    list_dirs = os.walk(input_folder)

    # revert_lm_char_dict = {v: k for k, v in lm_char_dict.items()}
    # revert_type_char_dict = {v: k for k, v in lm_type_dict.items()}
    
    dataset_text = list()
    dataset_type = list()

    for root, dirs, files in list_dirs:
        for file in tqdm(files):

            if file in complicated_list:
                continue

            with open(os.path.join(root, file), 'r') as fin:
                for line in tqdm(fin):
                    line = line.rstrip()

                    for char in line:
                        # ori_char = char

                        if char in lm_char_dict:
                            dataset_text.append(lm_char_dict[char])
                            dataset_type.append(lm_char_type_dict[char])
                        elif char.isupper() and char.lower() in lm_char_dict:
                            dataset_text.append(lm_char_dict[char.lower()])
                            dataset_type.append(lm_type_dict['<up>'])
                        elif char in emoji.UNICODE_EMOJI:
                            dataset_text.append(lm_char_dict['<emoji>'])
                            dataset_type.append(lm_type_dict['<pun>'])
                        else:
                            char_list = unicodedata.normalize('NFKD', char)

                            for char_tup in char_list:
                                if char_tup in lm_char_dict:
                                    dataset_text.append(lm_char_dict[char_tup])
                                    dataset_type.append(lm_char_type_dict[char_tup])
                                elif char_tup.isupper():
                                    char_tup = char_tup.lower()
                                    if char_tup in lm_char_dict:
                                        dataset_text.append(lm_char_dict[char])
                                    else:
                                        dataset_text.append(lm_char_dict['<unk>'])
                                    dataset_type.append(lm_type_dict['<up>'])
                                elif char_tup.islower():
                                    dataset_text.append(lm_char_dict['<unk>'])
                                    dataset_type.append(lm_type_dict['<low>'])
                                elif char_tup.isdigit():
                                    dataset_text.append(lm_char_dict['<unk>'])
                                    dataset_type.append(lm_type_dict['<num>'])
                                else:
                                    dataset_text.append(lm_char_dict['<unk>'])
                                    dataset_type.append(lm_type_dict['<pun>'])
                        # print('{} : {}, {}'.format(ori_char, revert_lm_char_dict[dataset_text[-1]], revert_type_char_dict[dataset_type[-1]]))
                        
                    # set_trace()
                    dataset_text.append(lm_char_dict['<eof>'])
                    dataset_type.append(lm_type_dict['<pun>'])

                    if len(dataset_text) > 20000000:

                        with open(os.path.join(output_folder,'train_'+ str(range_ind) + '.pk'), 'wb') as fout:
                            pickle.dump({'text_array': dataset_text, 'type_array': dataset_type}, fout)

                        dataset_text = list()
                        dataset_type = list()

                        range_ind += 1

                        with open('tmp.json', 'w') as fout:
                            json.dump({'complicated_list': complicated_list, 'range_ind': range_ind}, fout)

            complicated_list.append(file)

    with open(os.path.join(output_folder,'train_'+ str(range_ind) + '.pk'), 'wb') as fout:
        pickle.dump({'text_array': dataset_text, 'type_array': dataset_type}, fout)

    return range_ind

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dict', default="./enriched_char_dict.json")
    parser.add_argument('--input_folder', default="./twitter_raw")
    parser.add_argument('--output_folder', default="./twitter_encoded")
    args = parser.parse_args()

    with open(args.input_dict, 'r') as fin:
        all_dict = json.load(fin)

        char_dict = all_dict['char_dict']
        lm_char_dict = all_dict['lm_char_dict']
        lm_type_dict = all_dict['lm_type_dict']
    # print(lm_char_dict)
    encode_dataset2file(lm_char_dict, lm_type_dict, args.input_folder, args.output_folder)
