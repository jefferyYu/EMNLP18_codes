# -*- coding: utf-8 -*-
# @Time    : 2017/1/12 15:25

from ec_data_topic_utils import Data
import argparse
import sys
import os
from ec_bilstm_kl_topic_self_att_dasts import BiLstm
import numpy as np

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

def gen_data(input_sents, word2id,max_length, max_topic_len):
    print("max_length:" + str(max_length)+ "max_topic_length:"+ str(max_topic_len))
    sentence = []
    sentence_weight = []
    sentence_length = []
    sentence_topic = []
    sentence_topic_weight = []
    sentence_topic_length = []

    for sent in input_sents:
            words = []
            weight = []
            topics = []
            topic_weight = []
            word_list = text_processor.pre_process_doc(sent)
            sent_length = len(word_list)
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
            sentence_topic.append(np.asarray(topics))
            sentence_topic_length.append(topic_len)
            sentence_topic_weight.append(np.asarray(topic_weight))
    input_array =np.asarray(sentence)
    weight_array=np.asarray(sentence_weight)
    length_array = np.asarray(sentence_length)
    topic_input_array = np.asarray(sentence_topic)
    topic_weight_array = np.asarray(sentence_topic_weight)
    topic_length_array = np.asarray(sentence_topic_length)
    
    return input_array, weight_array, length_array, topic_input_array, topic_weight_array, topic_length_array

def init_parameters():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-lr', '--learn_rate', help=" ", type=float,default=0.001) #originally is set to 0.001
    parser.add_argument('-ep', '--epochs',type=int,help="training epochs  ",default=1)
    parser.add_argument('-hs', '--hidden_size', help="hidden layer size", default=200)
    parser.add_argument('-ln', '--num_layers', help="stack lstm number", default=1)
    parser.add_argument('-wes', '--word_embed_size', help="word vect size",
            default=300)
    parser.add_argument('-ds', '--data_size', help="data size", default=0)
    parser.add_argument('-bs', '--batch_size', help=" ", default=60) #60
    parser.add_argument('-mn', '--model_name', help="model saved path",default='model')
    parser.add_argument('-md', '--mode', help="train or test",default='test')
    parser.add_argument('-dn', '--data_name', help="dataset name",default='SemEval18') #'SemEval18'
    args = parser.parse_args()
    print('current dataset is '+str(args.data_name))
    return args

def sent_preprocess(input_sentence, args):
    data = Data('./data/'+args.data_name+'data_sample.bin','./data/'+args.data_name+'vocab_sample.bin',
                            './data/'+args.data_name+'word_embed_weight_sample.bin',args.batch_size)
            
    input,weight,length,topic_input,topic_weight,topic_length = gen_data(input_sentence, data.word2id, \
                                                                     data.max_sent_len, \
                                                                     data.max_topic_len)
    return data,input,weight,length,topic_input,topic_weight,topic_length
    
class JointClassifier(object):
    def __init__(self):
        self.args = init_parameters()
        self.inputs = ['Let us go girls!! 🎀 GLEN A. WILSON HIGH SCHOOL']
        self.data,_,_,_,_,_,_= sent_preprocess(self.inputs, self.args)
        if not os.path.exists('./' + self.args.data_name + '_output/'):
                os.makedirs('./' + self.args.data_name + '_output/')
        self.model = BiLstm(self.args, self.data, ckpt_path='./' + self.args.data_name + '_output/')
        self.sess = self.model.restore_last_session()
    
    def predict(self, input_sentence):
        data,input,weight,length,topic_input,topic_weight,topic_length= sent_preprocess([input_sentence], self.args)
        sent_preds, sent_pred_probs, pred_scores, pred_prob_scores = \
            self.model.joint_predict(self.sess,input,weight,length,topic_input,topic_weight,topic_length)
        
        sentiment_list = ['neutral','positive','negative']
        emotion_list = ['anger','anticipation','disgust','fear','joy','love','optimism',\
                            'pessimism','sadness','surprise','trust']
        print('-'*50)
        print(input_sentence)
        for i in range(len(sent_preds)):
            sent = sent_preds[i]
            print('sentiment: '+ sentiment_list[sent])
            sent_prob = sent_pred_probs[i]
            print('neutral--'+ str(sent_prob[0])+' positive: '+ str(sent_prob[1])+ ' negative: '+ str(sent_prob[2]))
            emotion = pred_scores[i]
            emotion_str = ''
            for j in range(len(emotion)):
                if emotion[j] != 0:
                    emotion_str+=emotion_list[j]+', '
            print('emotion: '+emotion_str[:-2])
            emotion_prob = pred_prob_scores[i]
            emotion_prob_str = ''
            for k in range(len(emotion)):
                emotion_prob_str += emotion_list[k]+': '+str(emotion_prob[k])+ ', '
            print('emotion prob--'+ emotion_prob_str[:-2])

if __name__ == '__main__':
    a = JointClassifier()
    a.predict('They need to escape the war.')
    a.predict('Fuck this shit i quit at life for the time being lmfao')
    a.predict('Parade Wensday ITS GOINGGGGGG 🏀 🤘🏾 💪🏾 💯 #ALLINCLE')
    a.predict('From The Voice 😅 😅 😅')
    a.predict('We watched Infinity war.')
    a.predict('He went to the gay parade.')
    a.predict('We went to the gay parade. It was fun')








