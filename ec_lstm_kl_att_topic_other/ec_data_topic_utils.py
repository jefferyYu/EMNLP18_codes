# -*- coding: utf-8 -*-
# @Time    : 2017/1/12 16:04

import math
import pickle
import six

class Data(object):
        def __init__(self,data_path,vocab_path,pretrained,batch_size):
                self.batch_size = batch_size

                data, vocab ,pretrained= self.load_vocab_data(data_path,vocab_path,pretrained)
                self.train=data['train']
                self.valid=data['valid']
                self.test=data['test']
                self.train2=data['train2']
                self.valid2=data['valid2']
                self.test2=data['test2']
                self.word_size = len(vocab['word2id'])+1
                self.max_sent_len = vocab['max_sent_len']
                self.max_topic_len = vocab['max_topic_len']
                word2id = vocab['word2id']                
                #self.id2word = dict((v, k) for k, v in word2id.iteritems())
                self.id2word = {}
                for k, v in six.iteritems(word2id):
                    self.id2word[v]=k
                self.pretrained=pretrained

        def gen_batch(self,data,i):
                begin=i*self.batch_size
                data_size=data['input'].shape[0]
                end=(i+1)*self.batch_size
                if end>data_size:
                        end=data_size

                input=data['input'][begin:end]
                target = data['target'][begin:end]
                #target = data['label'][begin:end] #sigmoid: drop at least 3 points
                label = data['label'][begin:end,:-1]
                length=data['length'][begin:end]
                weight=data['weight'][begin:end]
                topic_input=data['topic_input'][begin:end]
                topic_length=data['topic_length'][begin:end]
                topic_weight=data['topic_weight'][begin:end]
                return input,target,label,weight,length,topic_input,topic_weight,topic_length

        def gen_sent_batch(self,data,i):
                begin=i*self.batch_size
                data_size=data['input'].shape[0]
                end=(i+1)*self.batch_size
                if end>data_size:
                        end=data_size

                input=data['input'][begin:end]
                target = data['target'][begin:end]
                #target = data['label'][begin:end] #sigmoid: drop at least 3 points
                length=data['length'][begin:end]
                weight=data['weight'][begin:end]
                topic_input=data['topic_input'][begin:end]
                topic_length=data['topic_length'][begin:end]
                topic_weight=data['topic_weight'][begin:end]
                return input,target,weight,length,topic_input,topic_weight,topic_length

        def load_vocab_data(self,data_path,vocab_path,pretrained):

                with open(data_path, 'rb') as fdata, open(vocab_path, 'rb') as fword2id:
                        data = pickle.load(fdata)
                        vocab = pickle.load(fword2id)

                with open(pretrained, 'rb') as fin:
                        pretrained = pickle.load(fin)

                return data, vocab, pretrained


        def gen_batch_num(self,data): #
                data_size = data['input'].shape[0]
                batch_num=math.ceil(data_size/float(self.batch_size))

                return int(batch_num)
