"""
.. module:: string-pre-pipeline
    :synopsis: implementation of strEncoder for string match and lm

.. moduleauthor:: Liyuan Liu
"""
import re
import sys
import json
import emoji
import logging
import unicodedata
import torch
import random

logger = logging.getLogger(__name__)

from betanlp.common.utils import iob_iobes
from betanlp.encoder.nltk_wrapper import NLTK_wrapper

from pytrie import PyTrie
from tqdm import tqdm

class strEncoder(object):
    """
    string encoder: convert string into list of one-hot tensors.
    """
    def __init__(self, arg):
        super(strEncoder, self).__init__()
        if type(arg) is not dict:
            arg = vars(arg)

        self.pipeline_dict = dict()
        self.label_pipeline = None
        self.build_pipelines(arg)

    def build_pipelines(self, arg):
        raise NotImplementedError

    def forward(self, x, y = None):
        output_dict = dict()

        for k, v in self.pipeline_dict.items():
            output_dict[k] = v(x)

        if y is not None and self.label_pipeline is not None:
            label = self.label_pipeline(y)

        return output_dict, label

    def __call__(self, x, y = None):
        return self.forward(x, y)

class strPipeline(object):
    """
    string encoder: convert string into list of one-hot tensors.
    """
    def __init__(self, arg):
        super(strPipeline, self).__init__()

        if type(arg) is not dict:
            arg = vars(arg)

        self.build_pipeline(arg)

    def build_pipeline(self, arg):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

class strOriRealTimePipeline(strPipeline):
    """
    string encoder: convert string without changing anything.
    """
    def build_pipeline(self, arg):
        return

    def forward(self, x):
        return [{'text': tup, 'len': len(tup)} for tup in x]

class strLMRealTimePipeline(strPipeline):
    """
    string encoder: convert string into list of one-hot tensors for language models.
    """
    def build_pipeline(self, arg):
        assert 'lm_dict' in arg
        try: 
            with open(arg['lm_dict'], 'r') as fin:
                tmp_dict = json.load(fin)
                self.lm_char_dict = tmp_dict['lm_char_dict']
                self.lm_type_dict = tmp_dict['lm_type_dict']
        except FileNotFoundError as err:
            logger.error('File not exist: {}'.format(arg['lm_dict']))
            raise

        self.lm_char_type_dict = dict()

        for k, v in self.lm_char_dict.items():
            if k.islower():
                self.lm_char_type_dict[k] = self.lm_type_dict['<low>']
            elif k.isdigit():
                self.lm_char_type_dict[k] = self.lm_type_dict['<num>']
            else:
                self.lm_char_type_dict[k] = self.lm_type_dict['<pun>']

    def forward(self, x):
        instances = list()

        for line in x:
            text = list()
            category = list()
            char_len = list()

            for char in line:

                if char in self.lm_char_dict:
                    text.append(self.lm_char_dict[char])
                    category.append(self.lm_char_type_dict[char])
                    char_len.append(1)
                elif char.isupper() and char.lower() in self.lm_char_dict:
                    text.append(self.lm_char_dict[char.lower()])
                    category.append(self.lm_type_dict['<up>'])
                    char_len.append(1)
                elif char in emoji.UNICODE_EMOJI:
                    text.append(self.lm_char_dict['<emoji>'])
                    category.append(self.lm_type_dict['<pun>'])
                    char_len.append(1)
                else:
                    char_list = unicodedata.normalize('NFKD', char)

                    if len(char_list) > 0:

                        for char_tup in char_list:

                            if char_tup in self.lm_char_dict:
                                text.append(self.lm_char_dict[char_tup])
                                category.append(self.lm_char_type_dict[char_tup])
                            elif char_tup.isupper():
                                category.append(self.lm_type_dict['<up>'])
                                char_tup = char_tup.lower()
                                if char_tup in self.lm_char_dict:
                                    text.append(self.lm_char_dict[char_tup])
                                else:
                                    text.append(self.lm_char_dict['<unk>'])
                            elif char_tup.islower():
                                category.append(self.lm_type_dict['<low>'])
                                text.append(self.lm_char_dict['<unk>'])
                            elif char_tup.isdigit():
                                category.append(self.lm_type_dict['<num>'])
                                text.append(self.lm_char_dict['<unk>'])
                            else:
                                category.append(self.lm_type_dict['<pun>'])
                                text.append(self.lm_char_dict['<unk>'])

                        char_len.append(len(char_list))

            assert (len(text) == sum(char_len))
            assert (len(category) == sum(char_len))

            instances.append({'text': text, 'type': category, 'len': char_len, 'tot_len': len(text)})

        return instances

#########################################
#### REMARK: ASCII Code only for Now ####
#########################################

class strSMRealTimePipeline(strPipeline):
    """
    string encoder: convert string into list of one-hot tensors for language models.
    """
    def build_pipeline(self, arg):
        assert 'lm_dict' in arg
        try: 
            with open(arg['lm_dict'], 'r') as fin:
                self.lm_char_dict = json.load(fin)['lm_char_dict']
        except FileNotFoundError as err:
            logger.error('File not exist: {}'.format(arg['lm_dict']))
            raise

        if 'tokenizer' not in arg or arg['tokenizer'] == 'Trie':
            self.trie = PyTrie()
            try:
                if '\t' == arg['embed_seperator']:
                    self.trie.insert_from_file(arg['embed'].encode(), ord('\t'), True)
                else:
                    assert ' ' == arg['embed_seperator']
                    self.trie.insert_from_file(arg['embed'].encode(), ord(' '), True)
            except FileNotFoundError as err:
                logger.error('File not exist: {}'.format(arg['embed']))
                raise
            self.forward = self.trie_forward

        elif arg['tokenizer'] == 'NLTK':
            logger.info('Using nltk...')
            self.nltk_wrapper = NLTK_wrapper(arg['embed'])
            self.forward = self.nltk_forward

        else:
            error_msg = 'Unrecognized Tokenizer: {}.'.format(arg['tokenizer'])
            logger.error(error_msg)
            raise Exception(error_msg)

        self.convert_lower = lambda x: x
        if 'lower' in arg and arg['lower']:
            self.convert_lower = lambda x: x.lower()

        self.one_mention = False
        if 'one_mention' in arg and arg['one_mention']:
            self.one_mention = True

        self.remove_ht = False
        if 'remove_ht' in arg and arg['remove_ht']:
            self.remove_ht = True

        self.replace_num = True
        if 'replace_num' in arg and not arg['replace_num']:
            self.replace_num = False

    def nltk_forward(self, x):    
        return self.nltk_wrapper.forward(x)

    def trie_forward(self, x):

        instances = list()

        for line in x:
            ori_len = len(line)

            start_ids, end_ids = range(0, len(line)), range(1, len(line) + 1)
            tmp_start_ids, tmp_end_ids = [-1] * len(line), [-1] * len(line)
            shift = 0
            def numrepl(matchobj):
                nonlocal shift, tmp_start_ids, tmp_end_ids
                rep_len = 10
                start_id = matchobj.start()
                end_id = matchobj.end()
                new_start = start_id + shift
                new_end = new_start + rep_len
                for tmp_ind in range(start_id, end_id):
                    tmp_start_ids[tmp_ind] = new_start
                    tmp_end_ids[tmp_ind] = new_end
                shift = shift + rep_len - (end_id - start_id)
                return '__number__'

            def after_care():
                nonlocal start_ids, end_ids
                new_start_ids, new_end_ids = [-1] * len(line), [-1] * len(line)
                shift = 0
                # for tmp_ind in range(len(tmp_start_ids)):
                tmp_ind = 0
                while (tmp_ind < len(tmp_start_ids)):
                    if -1 == tmp_start_ids[tmp_ind]:
                        new_start_ids[tmp_ind + shift] = start_ids[tmp_ind]
                        new_end_ids[tmp_ind + shift] = end_ids[tmp_ind]
                        tmp_ind += 1
                    else:
                        assert(tmp_ind+shift == tmp_start_ids[tmp_ind])
                        tmp_end_ind = tmp_ind + 1
                        while (tmp_end_ind < len(tmp_start_ids) and tmp_start_ids[tmp_end_ind] == tmp_start_ids[tmp_ind] and tmp_end_ids[tmp_end_ind] == tmp_end_ids[tmp_ind]):
                            tmp_end_ind += 1
                        for iter_tmp_ind in range(tmp_start_ids[tmp_ind], tmp_end_ids[tmp_ind]):
                            new_start_ids[iter_tmp_ind] = start_ids[tmp_ind]
                            new_end_ids[iter_tmp_ind] = end_ids[tmp_end_ind - 1]

                        shift = tmp_end_ids[tmp_ind] - tmp_end_ind
                        tmp_ind = tmp_end_ind

                start_ids = new_start_ids
                end_ids = new_end_ids

            if self.replace_num:
                line = re.sub(r'[-+]?\d+', numrepl, line)
                after_care()

            shift = 0
            tmp_start_ids, tmp_end_ids = [-1] * len(line), [-1] * len(line)
            def atrepl0(matchobj):
                nonlocal shift, tmp_start_ids, tmp_end_ids
                rep_len = 6
                start_id = matchobj.start()
                end_id = start_id + 1
                new_start = start_id + shift
                new_end = new_start + rep_len
                tmp_start_ids[start_id] = new_start
                tmp_end_ids[start_id] = new_end
                shift = shift + rep_len - 1
                return '__at__' + matchobj.group(0)[1:]
            def atrepl1(matchobj):
                nonlocal shift, tmp_start_ids, tmp_end_ids
                rep_len = 6
                start_id = matchobj.start()
                end_id = matchobj.end()
                new_start = start_id + shift
                new_end = new_start + rep_len
                for tmp_ind in range(start_id, end_id):
                    tmp_start_ids[tmp_ind] = new_start
                    tmp_end_ids[tmp_ind] = new_end
                shift = shift + rep_len - (end_id - start_id)
                return '__at__'
            if self.one_mention:
                line = re.sub(r'@[\da-zA-Z_]+', atrepl1, line)
            else:
                line = re.sub(r'@[\da-zA-Z_]+', atrepl0, line)
            after_care()

            shift = 0
            tmp_start_ids, tmp_end_ids = [-1] * len(line), [-1] * len(line)
            def htrepl0(matchobj):
                nonlocal shift, tmp_start_ids, tmp_end_ids
                rep_len = 6
                start_id = matchobj.start()
                end_id = start_id + 1
                new_start = start_id + shift
                new_end = new_start + rep_len
                tmp_start_ids[start_id] = new_start
                tmp_end_ids[start_id] = new_end
                shift = shift + rep_len - 1
                return '__ht__' + matchobj.group(0)[1:]
            def htrepl1(matchobj):
                nonlocal shift, tmp_start_ids, tmp_end_ids
                start_id = matchobj.start()
                end_id = start_id + 1
                new_start = start_id + shift
                new_end = new_start + 0
                tmp_start_ids[start_id] = new_start
                tmp_end_ids[start_id] = new_end
                shift = shift - 1
                return matchobj.group(0)[1:]
            if self.remove_ht:
                line = re.sub(r'#[\da-zA-Z_]+', htrepl1, line)
            else:
                line = re.sub(r'#[\da-zA-Z_]+', htrepl0, line)
            after_care()

            shift = 0
            tmp_start_ids, tmp_end_ids = [-1] * len(line), [-1] * len(line)
            def urlrepl(matchobj):
                nonlocal shift, tmp_start_ids, tmp_end_ids
                rep_len = 7
                start_id = matchobj.start()
                end_id = matchobj.end()
                new_start = start_id + shift
                new_end = new_start + rep_len
                for tmp_ind in range(start_id, end_id):
                    tmp_start_ids[tmp_ind] = new_start
                    tmp_end_ids[tmp_ind] = new_end
                shift = shift + rep_len - (end_id - start_id)
                return '__url__'
            line = re.sub(r'https?://[^\s]+', urlrepl, line)
            after_care()

            shift = 0
            tmp_start_ids, tmp_end_ids = [-1] * len(line), [-1] * len(line)
            line = re.sub(r'pic.twitter.com/[\da-zA-Z_]+', urlrepl, line)
            after_care()

            shift = 0
            new_start_ids, new_end_ids = list(), list()
            nline= b""
            for char, start_id, end_id in zip(line, start_ids, end_ids):

                if char in self.lm_char_dict:
                    nline += char.encode()
                    new_start_ids += [start_id]
                    new_end_ids += [end_id]
                elif char.isupper() and char.lower() in self.lm_char_dict:
                    nline += self.convert_lower(char).encode()
                    new_start_ids += [start_id]
                    new_end_ids += [end_id]
                elif char in emoji.UNICODE_EMOJI:
                    char = emoji.demojize(char)
                    char = char.replace(':', '___')
                    char = re.sub(r'[^\w\s]', '_', char)
                    nline += char.encode()
                    new_start_ids += [start_id] * len(char)
                    new_end_ids += [end_id] * len(char)
                else:
                    char_list = self.convert_lower(unicodedata.normalize('NFKD', char))
                    char_len_tmp = 0
                    for char_tup in char_list:
                        tmp_char = char_tup.encode()
                        nline += tmp_char
                        char_len_tmp += len(tmp_char)
                    new_start_ids += [start_id] * char_len_tmp
                    new_end_ids += [end_id] * char_len_tmp                    

            result = self.trie.match_whole_string(nline)
            assert(len(result) == len(new_start_ids))
            assert(len(result) == len(new_end_ids))

            start_id = 0
            new_result = [-1] * ori_len
            while start_id < len(result):
                end_id = start_id + 1
                while (end_id < len(result) and new_start_ids[end_id] == new_start_ids[start_id] and new_end_ids[end_id] == new_end_ids[start_id]):
                    end_id += 1
                
                tmp_res = result[start_id:end_id]
                most_count = max(set(tmp_res), key=tmp_res.count)
                for tmp_ind in range(new_start_ids[start_id], new_end_ids[start_id]):
                    new_result[tmp_ind] = most_count
                    if most_count == 0:
                        logger.debug('ID({}), Char({}), Code({}): Sen({})'.format(start_id, nline[start_id:end_id].decode(), nline[start_id: end_id], nline))

                start_id = end_id

            assert(new_end_ids[start_id - 1] == ori_len)
            instances.append({'text': new_result, 'len': ori_len})
        return instances

class strLabelRealTimePipeline(strPipeline):
    def build_pipeline(self, arg):

        self.generate_label_dict = arg.get('generate_label_dict', False)
        self.label_dict_file = arg['label_dict']
        # self.label_dict = {'<sof>': 1, '<eof>': 0}
        self.label_dict = dict()

        self.convert_to_iobes = arg.get('convert_to_iobes', False)

        if not self.generate_label_dict:
            try: 
                with open(arg['label_dict'], 'r') as fin:
                    self.label_dict = json.load(fin)
            except FileNotFoundError as err:
                logger.error('File not exist: {}'.format(arg['label_dict']))
                raise

    def forward(self, y):

        instances = list()

        for line in y:
            
            encoded = list()
            if self.convert_to_iobes:
                line = iob_iobes(line)

            for ins in line:
                if ins not in self.label_dict:
                    assert self.generate_label_dict
                    self.label_dict[ins] = len(self.label_dict)
                encoded.append(self.label_dict[ins])
            instances.append({'label': encoded, 'len': len(encoded)})

        if self.generate_label_dict:
            with open(self.label_dict_file, 'w') as fout:
                json.dump(self.label_dict, fout)

        return instances


class strFromFilePipeline(strPipeline):
    """
    string encoder: convert string into list of one-hot tensors.    
    """
    def __init__(self, arg, name):
        self.name = name
        super(strFromFilePipeline, self).__init__(arg)

    def build_pipeline(self, arg):
        assert 'processed_file' in arg
        try: 
            with open(arg['processed_file'], 'r') as fin:
                self.instances = json.load(fin)[self.name]
        except FileNotFoundError as err:
            logger.error('File not exist: {}'.format(arg['processed_file']))
            raise

        return len(self.instances)

    def forward(self, x):
        return [self.instances[ins] for ins in x]

    def size(self):
        return len(self.instances)

# class strLMFromFilePipeline(strFromFilePipeline):
#     """
#     string encoder: convert string into list of one-hot tensors for language models.
#     """
#     def __init__(self, arg):
#         super(strLMFromFilePipeline, self).__init__(arg, 'lm')

# class strSMFromFilePipeline(strFromFilePipeline):
#     """
#     string encoder: convert string into list of one-hot tensors for word embedding.
#     """
#     def __init__(self, arg):
#         super(strSMFromFilePipeline, self).__init__(arg, 'sm')

# class strLabelFromFilePipeline(strFromFilePipeline):
#     def __init__(self, arg):
#         super(strLabelFromFilePipeline, self).__init__(arg, 'label')

# class strOriFromFilePipeline(strFromFilePipeline):
#     def __init__(self, arg):
#         super(strOriFromFilePipeline, self).__init__(arg, 'ori')

class strRealTimeEncoderWrapper(strEncoder):

    def build_pipelines(self, arg):

        for key in arg['strEncoder']:

            if 'lm' in key:
                self.pipeline_dict[key] = strLMRealTimePipeline(arg['strEncoder'][key])

            elif 'sm' in key:
                self.pipeline_dict[key] = strSMRealTimePipeline(arg['strEncoder'][key])

            elif 'ori' in key:
                self.pipeline_dict[key] = strOriRealTimePipeline(arg['strEncoder'][key])

        if 'label' in arg['strEncoder']:
            self.label_pipeline = strLabelRealTimePipeline(arg['strEncoder']['label'])

class strFromFileEncoderWrapper(strEncoder):

    def __init__(self, arg, processed_file=None):

        if type(arg) is not dict:
            arg = vars(arg)

        if processed_file is not None:
            arg['processed_file'] = processed_file

        super(strFromFileEncoderWrapper, self).__init__(arg)

    def build_pipelines(self, arg):
        length = -1

        for key in arg['strEncoder']:
            # if 'lm' in key:
            #     arg['strEncoder'][key]['processed_file'] = arg['processed_file']
            #     logger.info('Building LM pipeline...')
            #     self.pipeline_dict[key] = strLMFromFilePipeline(arg['strEncoder'][key])
            #     length = self.pipeline_dict[key].size()

            # elif 'sm' in key:
            #     arg['strEncoder'][key]['processed_file'] = arg['processed_file']
            #     logger.info('Building SM pipeline...')
            #     self.pipeline_dict[key] = strSMFromFilePipeline(arg['strEncoder'][key])
            #     length = self.pipeline_dict[key].size()

            # elif 'ori' in key:
            #     arg['strEncoder'][key]['processed_file'] = arg['processed_file']
            #     logger.info('Building ORI pipeline...')
            #     self.pipeline_dict[key] = strOriFromFilePipeline(arg['strEncoder'][key])
            #     length = self.pipeline_dict[key].size()
            arg['strEncoder'][key]['processed_file'] = arg['processed_file']
            logger.info('Building {} pipeline...'.format(key))
            self.pipeline_dict[key] = strFromFilePipeline(arg['strEncoder'][key], key)
            length = self.pipeline_dict[key].size()

        if 'label' in arg['strEncoder']:
            arg['strEncoder']['label']['processed_file'] = arg['processed_file']
            logger.info('Building label pipeline...')
            self.label_pipeline = strFromFilePipeline(arg['strEncoder']['label'], 'label')
            assert (length > 0)
            assert (length == self.label_pipeline.size())

        self.index_length = length
        logger.info('All pipeline has been built for {}'.format(arg['processed_file']))


    def get_tqdm(self, device, batch_size, shuffle = True):

        return tqdm(self.reader(device, batch_size, shuffle), mininterval=2, total=self.index_length // batch_size, leave=False, file=sys.stdout, ncols=80)
    
    def reader(self, device, batch_size, shuffle = True):

        index = list(range(self.index_length))
        if shuffle:
            random.shuffle(index)

        cur_idx, end_idx = 0, batch_size
        while end_idx < self.index_length:
            yield self.forward(index[cur_idx: end_idx], index[cur_idx: end_idx])
            cur_idx = end_idx
            end_idx += batch_size
