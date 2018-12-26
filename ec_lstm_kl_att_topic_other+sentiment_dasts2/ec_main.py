# -*- coding: utf-8 -*-
# @Time    : 2017/1/12 15:25

from ec_data_topic_utils import Data
import argparse
import sys
import os
from ec_bilstm_kl_topic_self_att_dasts import BiLstm

def main(arguments):
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-ui', '--use_ilp', help="use ilp  or not  ", action='store_true', default=False)
	parser.add_argument('-lr', '--learn_rate', help=" ", type=float,default=0.001) #originally is set to 0.001
	parser.add_argument('-ep', '--epochs',type=int,help="training epochs  ",default=1)
	parser.add_argument('-hs', '--hidden_size', help="hidden layer size", default=200)
	parser.add_argument('-ln', '--num_layers', help="stack lstm number", default=1)
	parser.add_argument('-wes', '--word_embed_size', help="word vect size",
                default=300)
	parser.add_argument('-ds', '--data_size', help="data size", default=0)
	parser.add_argument('-bs', '--batch_size', help=" ", default=60) #60
	parser.add_argument('-mn', '--model_name', help="model saved path",default='model')
	parser.add_argument('-md', '--mode', help="train or test",default='train')
	parser.add_argument('-dn', '--data_name', help="dataset name",default='SemEval18') #'SemEval18'
	args = parser.parse_args(arguments)
	print('current dataset is '+str(args.data_name))

	data = Data('./data/'+args.data_name+'data_sample.bin','./data/'+args.data_name+'vocab_sample.bin',
				'./data/'+args.data_name+'word_embed_weight_sample.bin',args.batch_size)

	#data = Data('../ec_lstm_kl_att_topic_other+sentiment_wsdm/data/'+args.data_name+'data_sample.bin',
				#'../ec_lstm_kl_att_topic_other+sentiment_wsdm/data/'+args.data_name+'vocab_sample.bin',
				#'../ec_lstm_kl_att_topic_other+sentiment_wsdm/data/'+args.data_name+'word_embed_weight_sample.bin',
				#args.batch_size)

	#args.batch_size = 60
	if not os.path.exists('./' + args.data_name + '_output/'):
		os.makedirs('./' + args.data_name + '_output/')
	model = BiLstm(args, data, ckpt_path='./' + args.data_name + '_output/')

	if args.mode=='train':
		model.train(data)
		sess = model.restore_last_session()
		model.predict(data, sess)
	if args.mode=='test':
		sess = model.restore_last_session()
		model.predict(data, sess)

if __name__ == '__main__':
	#CUDA_VISIBLE_DEVICES = ""
	main(sys.argv[1:])








