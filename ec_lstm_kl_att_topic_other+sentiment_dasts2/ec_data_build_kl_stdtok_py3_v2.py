# -*- coding: utf-8 -*-
# @Time    : 2017/1/12 11:46
import pickle
import numpy as np
import os
import re
import six

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

text_processor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                           'time', 'url', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "allcaps", "elongated", "repeated",
                          'emphasis', 'censored'},
        fix_html=True,  # fix HTML tokens

        # corpus from which the word statistics are going to be used
        # for word segmentation
        segmenter="twitter",

        # corpus from which the word statistics are going to be used
        # for spell correction
        corrector="twitter",

        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
)

def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  """
  string = re.sub(r"[^A-Za-z0-9(),.!?#@\(\)\?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"\.", " . ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"#", " # ", string)
  string = re.sub(r"@", " @ ", string)
  string = re.sub(r"\(", " ( ", string)
  string = re.sub(r"\)", " ) ", string)
  string = re.sub(r"\?", " ? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower()

class DataBuild(object):
        def __init__(self):
                folder = '../data'
                #folder2 = '../data_twitter'
                dataset = 'SemEval18'

                self.train=folder+os.sep+dataset+'_train.txt'
                self.valid=folder+os.sep+dataset+'_dev.txt'
                self.test=folder+os.sep+dataset+'_test.txt'
                self.train2 = folder + os.sep + dataset + '_train_sent.txt'
                self.valid2 = folder + os.sep + dataset + '_dev_sent.txt'
                self.test2 = folder + os.sep + dataset + '_test_sent.txt'

                bin_dir = './data'
                if not os.path.exists(bin_dir):
                        os.mkdir(bin_dir)

                self.vocab_file=bin_dir+os.sep+dataset+'vocab_sample.bin'
                self.data_out=bin_dir+os.sep+dataset+'data_sample.bin'
                #self.embedding_path = '/home/jfyu/torch/stanford_treelstm-master/data/glove/glove.twitter.27B.200d.txt'
                #self.embedding_path = '../../glove.6B.50d.txt'         
                self.embedding_path = '../../glove.840B.300d.txt'               
                #self.embedding_path = '/home/jfyu/torch/stanford_treelstm-master/data/glove/glove.840B.300d.txt'
                #self.embedding_path = '/home/jfyu/torch/stanford_treelstm-master/data/glove/datastories.twitter.300d.txt'
                self.word_embed_size=300
                self.weight_path=bin_dir+os.sep+dataset+'word_embed_weight_sample.bin'

        def gen_vocab_data(self):
                file_list = [self.train2, self.train, self.valid, self.test]
                word2id,mlength,mtlength=self.gen_vocab(file_list)
                data=self.gen_data(word2id,mlength,mtlength)
                vocab={'word2id':word2id, 'max_sent_len':mlength, 'max_topic_len':mtlength}
                with open(self.vocab_file,'wb') as fout:
                        #pickle.dump(vocab,fout,protocol=2)
                        pickle.dump(vocab,fout)
                with open(self.data_out,'wb') as fout:
                        #pickle.dump(data,fout,protocol=2)
                        pickle.dump(data,fout)
                if os.path.exists(self.embedding_path):
                        embedding_weight=self.gen_trained_word_embedding(word2id)
                        with open(self.weight_path,'wb') as fout:
                                #pickle.dump(embedding_weight,fout,protocol=2)
                                pickle.dump(embedding_weight,fout)
                        print('weight genenrated')
                else:
                        print('pretrained word embedding weight should exist at:'+self.embedding_path)

                print("data and vocab built")

        def gen_vocab(self, file_list):
                max_sent_len = 0
                max_topic_len = 0
                word_to_idx = {}
                word_cnt_dict = {}

                # Starts at 1 for padding, position 0 is for padding
                idx = 1
                word_to_idx['UNKUNK'] = idx
                idx += 1
                word_to_idx['<hashtag>'] = idx
                idx += 1
                word_to_idx['</hashtag>'] = idx # </hashtag> will be indexed at 3
                idx += 1

                for filename in file_list:
                        if filename != self.train2:
                                fin_list = open(filename, "r", encoding='UTF-8').readlines()
                                for line in fin_list[1:]:
                                        line_list = line.strip().split('\t')
                                        sent = line_list[1]
                                        word_list = text_processor.pre_process_doc(sent)
                                        #print(word_list)
                                        #word_list = clean_str(sent).split()
                                        sent_len = len(word_list)
                                        topic_list = []
                                        begin = 0
                                        for i in range(len(word_list)):
                                                word = word_list[i]
                                                if word == '<hashtag>':
                                                        begin = i
                                                elif word == '</hashtag>':
                                                        end = i
                                                        for wd in range(begin + 1, end):
                                                                topic_word = word_list[wd]
                                                                topic_list.append(topic_word)
                                                elif word not in word_cnt_dict:
                                                        word_cnt_dict[word] = 1
                                                elif word in word_cnt_dict:
                                                        word_cnt_dict[word] += 1
                                        #print(topic_list)
                                        max_sent_len = max(max_sent_len, sent_len)
                                        max_topic_len = max(max_topic_len, len(topic_list))
                        else:
                                fin_list = open(filename, "r", encoding='UTF-8').readlines()
                                for line in fin_list[1:]:
                                        line_list = line.strip().split('\t')
                                        sent = line_list[1]
                                        word_list = text_processor.pre_process_doc(sent)
                                        #print(word_list)
                                        # word_list = clean_str(sent).split()
                                        sent_len = len(word_list)
                                        topic_list = []
                                        begin = 0
                                        for i in range(len(word_list)):
                                                word = word_list[i]
                                                if word == '<hashtag>':
                                                        begin = i
                                                elif word == '</hashtag>':
                                                        end = i
                                                        for wd in range(begin + 1, end):
                                                                topic_word = word_list[wd]
                                                                topic_list.append(topic_word)
                                                elif word not in word_cnt_dict:
                                                        word_cnt_dict[word] = 1
                                                elif word in word_cnt_dict:
                                                        word_cnt_dict[word] += 1
                                        #print(topic_list)
                                        max_sent_len = max(max_sent_len, sent_len)
                                        max_topic_len = max(max_topic_len, len(topic_list))

                for word, freq in six.iteritems(word_cnt_dict):
                    if freq > 1:
                        word_to_idx[word] = idx
                        idx += 1
                print('vocab size = %.3f' % (idx))
                return word_to_idx, max_sent_len, max_topic_len


        def gen_data(self,word2id,max_length, max_topic_len):
                print("max_length:" + str(max_length)+ "max_topic_length:"+ str(max_topic_len))
                train,valid,test=dict(),dict(),dict()
                for infile,out in zip([self.valid,self.train,self.test],[valid,train,test]):
                        sentence = []
                        sentence_tag = []
                        sentence_label = []
                        sentence_weight = []
                        sentence_length = []
                        sentence_topic = []
                        sentence_topic_weight = []
                        sentence_topic_length = []

                        fin_list = open(infile, 'r', encoding='UTF-8').readlines()
                        for line in fin_list[1:]:
                                words = []
                                weight = []
                                topics = []
                                topic_weight = []
                                line_list = line.strip().split('\t')
                                sent = line_list[1]
                                tag_list = line_list[2:]
                                sent_tag = []
                                sent_label = []
                                count = 0
                                for tag in tag_list:
                                        count += int(tag)
                                        sent_label.append(int(tag))
                                if count == 0:
                                        for tag in tag_list:
                                                sent_tag.append(0)
                                        sent_label.append(1)
                                        sent_tag.append(1)
                                else:
                                        for tag in tag_list:
                                                sent_tag.append(float(tag)/float(count))
                                        sent_label.append(0)
                                        sent_tag.append(0)
                                #print(sent)
                                ##filt_sent = clean_str(sent)
                                ##print(filt_sent)
                                ##word_list = filt_sent.split()
                                word_list = text_processor.pre_process_doc(sent)
                                #print(word_list)
                                sent_length = len(word_list)
                                #print(sent_tag)
                                topic_list = []
                                begin = 0
                                for i in range(len(word_list)):
                                        word = word_list[i]
                                        if word == '<hashtag>':
                                                begin = i
                                                words.append(word2id[word])
                                                weight.append(1)
                                        elif word == '</hashtag>':
                                                end = i
                                                words.append(word2id[word])
                                                weight.append(1)
                                                for wd in range(begin + 1, end):
                                                        topic_word = word_list[wd]
                                                        topic_list.append(topic_word)
                                                        if topic_word in word2id:
                                                                topics.append(word2id[topic_word])
                                                        else:
                                                                topics.append(1)
                                                        topic_weight.append(1)
                                        else:
                                                if word in word2id:
                                                        words.append(word2id[word])
                                                else:
                                                        words.append(1)                                         
                                                weight.append(1)
                                topic_len = len(topics)
                                for _ in range(max_length - sent_length):
                                        words.append(0)
                                        weight.append(0)
                                if topic_len == 0:
                                        topics.append(0)
                                        topic_len = 1
                                        topic_weight.append(1)
                                for _ in range(max_topic_len - topic_len):
                                        topics.append(0)
                                        topic_weight.append(0)
                                #print(topic_list)
                                #print(topic_weight)
                                #print(topic_len)
                                if len(words)!= max_length:
                                        print('supererror')
                                sentence.append(np.asarray(words))
                                sentence_weight.append(np.asarray(weight))
                                sentence_length.append(sent_length)
                                sentence_tag.append(np.asarray(sent_tag))
                                sentence_label.append(np.asarray(sent_label))
                                sentence_topic.append(np.asarray(topics))
                                sentence_topic_length.append(topic_len)
                                sentence_topic_weight.append(np.asarray(topic_weight))
                        input_array =np.asarray(sentence)
                        target_array=np.asarray(sentence_tag)
                        weight_array=np.asarray(sentence_weight)
                        length_array = np.asarray(sentence_length)
                        label_array = np.asarray(sentence_label)
                        topic_input_array = np.asarray(sentence_topic)
                        topic_weight_array = np.asarray(sentence_topic_weight)
                        topic_length_array = np.asarray(sentence_topic_length)

                        if infile == self.train:
                                N = input_array.shape[0]
                                perm = np.random.permutation(N)
                                out['input'] = input_array[perm]
                                out['target'] = target_array[perm]
                                out['weight'] = weight_array[perm]
                                out['length'] = length_array[perm]
                                out['label'] = label_array[perm]
                                out['topic_input'] = topic_input_array[perm]
                                out['topic_weight'] = topic_weight_array[perm]
                                out['topic_length'] = topic_length_array[perm]
                                print('after shuffle, total sentence number:' + str(len(sentence)))
                        else:
                                out['input'] = input_array
                                out['target'] = target_array
                                out['weight'] = weight_array
                                out['length'] = length_array
                                out['label'] = label_array
                                out['topic_input'] = topic_input_array
                                out['topic_weight'] = topic_weight_array
                                out['topic_length'] = topic_length_array
                                print('total sentence number:' + str(len(sentence)))

                train2, valid2, test2 = dict(), dict(), dict()
                for infile,out in zip([self.valid2,self.train2,self.test2],[valid2,train2,test2]):
                        sentence = []
                        sentence_tag = []
                        sentence_weight = []
                        sentence_length = []
                        sentence_topic = []
                        sentence_topic_weight = []
                        sentence_topic_length = []

                        fin_list = open(infile, 'r', encoding='UTF-8').readlines()
                        for line in fin_list[1:]:
                                words = []
                                weight = []
                                topics = []
                                topic_weight = []
                                line_list = line.strip().split('\t')
                                sent = line_list[1]
                                sent_tag = int(line_list[2])

                                #print(sent)
                                ##filt_sent = clean_str(sent)
                                ##print(filt_sent)
                                ##word_list = filt_sent.split()
                                word_list = text_processor.pre_process_doc(sent)
                                #print(word_list)
                                sent_length = len(word_list)
                                #print(sent_tag)
                                topic_list = []
                                begin = 0
                                for i in range(len(word_list)):
                                        word = word_list[i]
                                        if word == '<hashtag>':
                                                begin = i
                                                words.append(word2id[word])
                                                weight.append(1)
                                        elif word == '</hashtag>':
                                                end = i
                                                words.append(word2id[word])
                                                weight.append(1)
                                                for wd in range(begin + 1, end):
                                                        topic_word = word_list[wd]
                                                        topic_list.append(topic_word)
                                                        if topic_word in word2id:
                                                                topics.append(word2id[topic_word])
                                                        else:
                                                                topics.append(1)
                                                        topic_weight.append(1)
                                        else:   
                                                if word in word2id:
                                                        words.append(word2id[word])
                                                else:
                                                        words.append(1)
                                                weight.append(1)
                                topic_len = len(topics)
                                for _ in range(max_length - sent_length):
                                        words.append(0)
                                        weight.append(0)
                                if topic_len == 0:
                                        topics.append(0)
                                        topic_len = 1
                                        topic_weight.append(1)
                                for _ in range(max_topic_len - topic_len):
                                        topics.append(0)
                                        topic_weight.append(0)
                                #print(topic_list)
                                #print(topic_weight)
                                #print(topic_len)
                                if len(words)!= max_length:
                                        print('supererror')
                                sentence.append(np.asarray(words))
                                sentence_weight.append(np.asarray(weight))
                                sentence_length.append(sent_length)
                                sentence_tag.append(sent_tag)
                                sentence_topic.append(np.asarray(topics))
                                sentence_topic_length.append(topic_len)
                                sentence_topic_weight.append(np.asarray(topic_weight))
                        input_array =np.asarray(sentence)
                        target_array=np.asarray(sentence_tag)
                        weight_array=np.asarray(sentence_weight)
                        length_array = np.asarray(sentence_length)
                        topic_input_array = np.asarray(sentence_topic)
                        topic_weight_array = np.asarray(sentence_topic_weight)
                        topic_length_array = np.asarray(sentence_topic_length)

                        if infile == self.train2:
                                N = input_array.shape[0]
                                perm = np.random.permutation(N)
                                out['input'] = input_array[perm]
                                out['target'] = target_array[perm]
                                out['weight'] = weight_array[perm]
                                out['length'] = length_array[perm]
                                out['topic_input'] = topic_input_array[perm]
                                out['topic_weight'] = topic_weight_array[perm]
                                out['topic_length'] = topic_length_array[perm]
                                print('after shuffle, total sentence number:' + str(len(sentence)))
                        else:
                                out['input'] = input_array
                                out['target'] = target_array
                                out['weight'] = weight_array
                                out['length'] = length_array
                                out['topic_input'] = topic_input_array
                                out['topic_weight'] = topic_weight_array
                                out['topic_length'] = topic_length_array
                                print('total sentence number:' + str(len(sentence)))

                data=dict()
                data['train']=train
                data['valid']=valid
                data['test']=test
                data['train2']=train2
                data['valid2']=valid2
                data['test2']=test2

                #total,unk=self.unk_statistic(gtest)
                #print('test total unk num:')
                #print(total,unk)
                return data

        def gen_trained_word_embedding(self,word2id): # glove embedding
                embeddings_index = {}
                #f = open(self.embedding_path, 'r', encoding='UTF-8')
                f = open(self.embedding_path, 'r', encoding='UTF-8')
                for line in f:
                        values = line.split()
                        word = values[0]
                        try:
                                coefs = np.asarray(values[1:], dtype='float32')
                        except ValueError:
                                print('word embedding small error')
                        embeddings_index[word] = coefs
                f.close()
                embedding_matrix = np.random.uniform(-0.25, 0.25, (len(word2id)+1, self.word_embed_size))
                embedding_matrix[0] = 0
                # embedding_matrix = np.zeros((len(self.word2id), self.word_embed_size))
                vocab_size=len(word2id)+1
                pretrained_size=0
                for word, i in word2id.items():
                        embedding_vector = embeddings_index.get(word)
                        if embedding_vector is not None:
                                # words not found in embedding index will be all-zeros.
                                pretrained_size+=1
                                #embedding_matrix[i+1] = embedding_vector # this place has some problems.
                                embedding_matrix[i] = embedding_vector  # this place has some problems.

                print('vocab size:%d\t pretrained size:%d' %(vocab_size,pretrained_size))

                return embedding_matrix

if __name__=='__main__':

        db=DataBuild()
        db.gen_vocab_data()
