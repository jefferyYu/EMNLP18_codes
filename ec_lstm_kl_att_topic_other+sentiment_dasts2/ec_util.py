# -*- coding: utf-8 -*-
# @Time    : 2017/1/22 14:52
class Util(object):
	def __init__(self):
		pass

	def caculate_length(self, fathers):
		dep_length = []
		for i in range(len(fathers)):
			if i == 0:
				dep = 0
				dep_length.append(dep)
				continue  # for bos
			dep = 1
			fa = fathers[i]
			while fa != 0:
				dep += 1
				fa = fathers[fa]
			dep_length.append(dep)
		maxl = max(dep_length)
		dep_length = [dl / float(maxl) for dl in dep_length]
		return dep_length

	def get_parentlist(self, fathers, rels):
		saved_type=[u'nsubj', u'dobj']
		saved = []
		for i, type in enumerate(rels):
			if type in saved_type:
				fid = fathers[i]-1
				if fid not in saved:
					saved.append(fid)
		return saved

