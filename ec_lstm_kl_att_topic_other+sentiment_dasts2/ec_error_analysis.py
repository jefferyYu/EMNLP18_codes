# -*- coding: utf-8 -*-
# @Time    : 2017/1/12 11:46
import operator
import os

class DataBuild(object):
	def __init__(self):
		folder = '../data'
		dataset = 'SemEval18'

		self.train=folder+os.sep+dataset+'_train.txt'
		self.valid=folder+os.sep+dataset+'_dev.txt'
		self.test=folder+os.sep+dataset+'_test.txt'
		self.pred = './pred_label.txt'
		self.label = './true_label.txt'
		self.pred_bas = '../ec_lstm_kl_att_topic_other/pred_label.txt'

	def gen_vocab_data(self):
		self.gen_data()

	def gen_data(self):
		#for infile in [self.valid,self.train,self.test, self.pred, self.label]:
		pred_list = []
		true_list = []
		pred_bas_list = []
		sent_list = []
		emotion_label_list = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism',
							  'sadness', 'surprise', 'trust']
		for infile in [self.test]:
			fin_list = open(infile, 'r').readlines()
			i =1
			label_content = []
			if infile == self.train:
				print('#################train####################')
				label_content = fin_list[1:]
				#print(fin_list[i])
			elif infile == self.valid:
				print('#################valid####################')
				label_content = fin_list[1:]
				#print(fin_list[i])
			elif infile == self.test:
				print('#################test####################')
				label_content = fin_list[1:]
				#print(fin_list[i])

			for line in label_content:
				line_list = line.strip().split('\t')
				sent = line_list[1]
				sent_list.append(sent)
		for infile, label_list in zip([self.pred, self.label, self.pred_bas], [pred_list, true_list, pred_bas_list]):
			label_dict = {}

			fin_list = open(infile, 'r').readlines()
			i =1
			if infile == self.train:
				print('#################train####################')
				label_content = fin_list[1:]
				#print(fin_list[i])
			elif infile == self.valid:
				print('#################valid####################')
				label_content = fin_list[1:]
				#print(fin_list[i])
			elif infile == self.test:
				print('#################test####################')
				label_content = fin_list[1:]
				#print(fin_list[i])
			elif infile == self.pred:
				print('#################predict_label####################')
				label_content = fin_list[1:]
				#print(fin_list[i-1])
			else:
				print('#################test_ground_truth#####################')
				label_content = fin_list[1:]
				#print(fin_list[i-1])

			emotion_dict = dict(zip(range(0,11), [0,0,0,0,0,0,0,0,0,0,0]))


			for line in label_content:
				line_list = line.strip().split('\t')
				sent = line_list[1]
				tag_list = line_list[2:]
				label_list.append(tag_list)
				label_str = ' '.join(tag_list)
				if label_str in label_dict:
					label_dict[label_str] +=1
				else:
					label_dict[label_str] = 1
				for idx in range(len(tag_list)):
					tag = tag_list[idx]
					if tag == '1':
						emotion_dict[idx] += 1

			label_type = label_dict.keys()
			print('label_size: '+ str(len(label_type)))
			print(emotion_dict)
			'''
			Other = '0 0 0 0 0 0 0 0 0 0 0'
			if Other in label_dict:
				print(label_dict[Other])
			else:
				print('NO')
			'''
			'''
			count = 0
			for key, value in label_dict.iteritems():
				if value > 10:
					count += 1
			print('label with more than 10 size: '+ str(count))
			'''
			#sorted_x = sorted(label_dict.items(), key=operator.itemgetter(1), reverse= True)
			#for key in sorted_x:
				#print(key)
		count = 0
		max_label_num_p = 0
		max_label_num_t = 0
		fout = open('error_analysis.txt', 'w')
		for i in range(len(pred_list)):
			p = pred_list[i]
			t = true_list[i]
			b = pred_bas_list[i]
			#if p == t:
				#count += 1
				#print(i)
			#else:
			if p == t and b!= t:
				fout.write('error index: ' + str(i)+'\n')
				fout.write(sent_list[i]+'\n')
				#print(p)
				pred_emotion = []
				for j in range(len(p)):
					em = p[j]
					if em == '1':
						pred_emotion.append(emotion_label_list[j])
				fout.write('pred_label: '+' '.join(pred_emotion)+'\n')
				pred_emotion = []
				for j in range(len(b)):
					em = b[j]
					if em == '1':
						pred_emotion.append(emotion_label_list[j])
				fout.write('bas_pred_label: '+' '.join(pred_emotion)+'\n')
				#print(t)
				true_emotion = []
				for j in range(len(t)):
					em = t[j]
					if em == '1':
						true_emotion.append(emotion_label_list[j])
				fout.write('true_label: '+' '.join(true_emotion)+'\n')
		fout.close()
		'''
			#this is for judging the number of max label in a single instance
			else:
				c1 = 0
				for j in range(len(p)):
					em = p[j]
					if em == '1':
						c1+=1
				c2 = 0
				for j in range(len(t)):
					em = t[j]
					if em == '1':
						c2+=1
				max_label_num_p = max(max_label_num_p, c1)
				max_label_num_t = max(max_label_num_t, c2)

		'''
		print(count)
		print(max_label_num_p)
		print(max_label_num_t)

if __name__=='__main__':

	db=DataBuild()
	db.gen_vocab_data()
