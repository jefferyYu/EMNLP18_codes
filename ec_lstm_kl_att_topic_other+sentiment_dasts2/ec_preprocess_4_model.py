# -*- coding: utf-8 -*-
# @Time    : 2017/1/12 11:46
import operator
import os

class DataBuild(object):
	def __init__(self):
		folder = '../data'
		folder2 = '../data_twitter'
		dataset = 'SemEval18'
		dataset2 = 'twitter-2016'

		self.train=folder+os.sep+dataset+'_train.txt'
		self.valid=folder+os.sep+dataset+'_dev.txt'
		self.test=folder+os.sep+dataset+'_test.txt'
		self.train2 = folder2 + os.sep + dataset2 + '_all.txt'
		self.senttrain = folder+os.sep+dataset+'_train_sent_emo.txt'
		self.sentvalid = folder+os.sep+dataset+'_dev_sent.txt'
		self.senttest = folder+os.sep+dataset+'_test_sent.txt'
		self.senttrain2 = folder+os.sep+dataset+'_train_sent.txt'


	def gen_vocab_data(self):
		self.gen_data()

	def gen_data(self):
		for infile in [self.valid,self.train,self.test,self.train2]:
			label_dict = {}
			em_dict = {'anger':0, 'anticipation':1, 'disgust':2, 'fear':3, 'joy':4, 'love':5,
								  'optimism':6, 'pessimism':7, 'sadness':8, 'surprise':9, 'trust':10}
			#rev_em_dict = {v: k for k, v in em_dict.iteritems()}
			pos_label_list = ['anticipation', 'joy', 'love', 'optimism', 'surprise', 'trust']
			str_pos_label = '\t'.join(pos_label_list)
			neg_label_list = ['anger','pessimism','disgust', 'fear', 'sadness']
			str_neg_label = '\t'.join(neg_label_list)
			pos_num_list = []
			for label in pos_label_list:
				pos_num_list.append(em_dict[label])
			neg_num_list = []
			for label in neg_label_list:
				neg_num_list.append(em_dict[label])
			label_num_list = [0,1,2,3,4,5,6,7,8,9,10]

			fin_list = open(infile, 'r', encoding = 'UTF-8').readlines()
			label_content = fin_list[1:]
			#i =1
			if infile == self.train:
				print('#################train####################')
				fout1 = open(self.senttrain,'w')
				#print(fin_list[i])
			elif infile == self.valid:
				print('#################valid####################')
				label_content = fin_list[1:]
				fout1 = open(self.sentvalid,'w')
				#print(fin_list[i])
			elif infile == self.test:
				print('#################test####################')
				label_content = fin_list[1:]
				fout1 = open(self.senttest,'w')
				#print(fin_list[i])
			else:
				print('#################twitter train####################')
				label_content = fin_list
				fout1 = open(self.senttrain2, 'w')
			# print(fin_list[i])
			fout1.write('ID\tTweet\tSentiment\n')

			if infile != self.train2:
				for line in label_content:
					pos_cnt = 0
					neg_cnt = 0
					line_list = line.strip().split('\t')
					tweetid = line_list[0]
					sent = line_list[1]
					tag_list = line_list[2:]
					#label_list.append(tag_list)
					for idx in range(len(tag_list)):
						tag = tag_list[idx]
						if tag == '1' and idx in pos_num_list:
							pos_cnt += 1
						elif tag== '1' and idx in neg_num_list:
							neg_cnt += 1

					if pos_cnt >0 and neg_cnt>0: #mix = 1
						if infile == self.train:
							continue
						else:
							fout1.write(tweetid + '\t' + sent + '\t' + '1' + '\n')
					elif pos_cnt >0 and neg_cnt==0: # pos=2
						fout1.write(tweetid + '\t' + sent + '\t' + '1'+'\n')
					elif pos_cnt ==0 and neg_cnt>0: # neg=3
						fout1.write(tweetid + '\t' + sent + '\t' + '2'+'\n')
					else: #neutral = 0
						fout1.write(tweetid + '\t' + sent + '\t' + '0'+'\n')
			else:
				for line in label_content:
					line_list = line.strip().split('\t')
					if len(line_list) < 3:
						print(line_list)
						continue
					else:
						tweetid = line_list[0]
						sent = line_list[2]
						tag = line_list[1]

					if tag == 'positive':  # pos=2
						fout1.write(tweetid + '\t' + sent + '\t' + '1' + '\n')
					elif tag == 'negative':  # neg=3
						fout1.write(tweetid + '\t' + sent + '\t' + '2' + '\n')
					elif tag == 'neutral': # no = 0
						fout1.write(tweetid + '\t' + sent + '\t' + '0' + '\n')
					else:
						print('error')

			fout1.close()


if __name__=='__main__':

	db=DataBuild()
	db.gen_vocab_data()
