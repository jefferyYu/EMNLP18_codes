import tensorflow as tf
import numpy as np
import sys
import time
from ec_util import Util
import tensorflow.contrib.rnn as RNNCell
from sklearn.metrics import f1_score, jaccard_similarity_score
#from ilp import Ilp
import os

def kl_loss(d_tgt, d_pred):
        return tf.reduce_sum(d_tgt*tf.log(tf.maximum(1e-8, d_tgt/d_pred)))

def add_gradient_noise(t, stddev=1e-3, name=None):
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.cast(tf.convert_to_tensor(t, name="t"), tf.float32)
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

def attention_mask(inputs, hidden_size, mask, return_alphas=False):
        #inputs matrix: batch_size, sequence length, hidden_size
        sequence_length = inputs.shape[1].value # the length of sequences processed in the antecedent RNN layer
        #hidden_size = inputs.shape[2].value # hidden size of the RNN layer

        # Attention mechanism - one
        W_omega = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
        v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))

        # Attention mechanism - two
        #u_omega = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
        #v = tf.reshape(inputs, [-1, hidden_size])

        vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
        vu = tf.reshape(vu, [-1, sequence_length]) # batch_size * sequence_length
        score_mask = tf.cast(mask, tf.bool) # batch_size * sequence_length
        score_mask_values = -1e8 * tf.ones_like(vu)
        scores = tf.where(score_mask, vu, score_mask_values)
        scores = tf.nn.softmax(scores)
        score_mask_values = 0 * tf.ones_like(scores)
        alphas = tf.where(score_mask, scores, score_mask_values)

        # Output of Bi-RNN is reduced with attention vector
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

        if not return_alphas:
                return output
        else:
                return output, alphas

def self_attention_mask(inputs, hidden_size, mask, sent_rep, return_alphas=False):
        #inputs matrix: batch_size, sequence length, hidden_size
        # sent_rep: batch_size, hidden size
        sequence_length = inputs.shape[1].value # the length of sequences processed in the antecedent RNN layer
        #hidden_size = inputs.shape[2].value # hidden size of the RNN layer
        #sent_reps = tf.tile(sent_rep, [1,sequence_length])
        sent_reps = tf.tile(sent_rep, [1,sequence_length])
        print(sent_reps.shape)

        # Attention mechanism - one
        W_omega = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.1))
        W_s = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
        v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) +
                                tf.matmul(tf.reshape(sent_reps, [-1, hidden_size]), W_s) + tf.reshape(b_omega, [1, -1]))

        # Attention mechanism - two
        #u_omega = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
        #v = tf.reshape(inputs, [-1, hidden_size])

        vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
        vu = tf.reshape(vu, [-1, sequence_length]) # batch_size * sequence_length
        score_mask = tf.cast(mask, tf.bool) # batch_size * sequence_length
        score_mask_values = -1e8 * tf.ones_like(vu)
        scores = tf.where(score_mask, vu, score_mask_values)
        scores = tf.nn.softmax(scores)
        score_mask_values = 0 * tf.ones_like(scores)
        alphas = tf.where(score_mask, scores, score_mask_values)

        # Output of Bi-RNN is reduced with attention vector
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

        if not return_alphas:
                return output
        else:
                return output, alphas

def self_topic_attention_mask(inputs, hidden_size, mask, sent_rep, top_rep, return_alphas=False):
        #inputs matrix: batch_size, sequence length, hidden_size
        # sent_rep: batch_size, hidden_size
        sequence_length = inputs.shape[1].value # the length of sequences processed in the antecedent RNN layer
        #hidden_size = inputs.shape[2].value # hidden size of the RNN layer
        #sent_reps = tf.tile(sent_rep, [1,sequence_length])
        sent_reps = tf.tile(sent_rep, [1,sequence_length]) # batch_size, sequence_len*hidden_size
        print(sent_reps.shape)
        top_reps = tf.tile(top_rep, [1,sequence_length])

        # Attention mechanism - one
        W_omega = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.1))
        W_s = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.1))
        W_t = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
        v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) +
                                tf.matmul(tf.reshape(sent_reps, [-1, hidden_size]), W_s) +
                                tf.matmul(tf.reshape(top_reps, [-1, hidden_size]), W_t) + tf.reshape(b_omega, [1, -1]))

        # Attention mechanism - two
        #u_omega = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
        #v = tf.reshape(inputs, [-1, hidden_size])

        vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
        vu = tf.reshape(vu, [-1, sequence_length]) # batch_size * sequence_length
        score_mask = tf.cast(mask, tf.bool) # batch_size * sequence_length
        score_mask_values = -1e8 * tf.ones_like(vu)
        scores = tf.where(score_mask, vu, score_mask_values)
        scores = tf.nn.softmax(scores)
        score_mask_values = 0 * tf.ones_like(scores)
        alphas = tf.where(score_mask, scores, score_mask_values)

        # Output of Bi-RNN is reduced with attention vector
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

        if not return_alphas:
                return output
        else:
                return output, alphas

def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res

def max_pool_function(inputs, mask):
        # inputs: batch_size * sequence_length * hidden_size
        # mask: batch_size * sequence_length
        #mask = tf.tile(mask, [inputs.shape[2].value, 1])
        inputs_trans = tf.transpose(inputs, [0,2,1]) #batch_size * hidden_size * sequence_length
        mask = tf.tile(mask, [1,inputs.shape[2].value]) # batch_size, hidden_size * sequence_length
        #print('mask')
        #print(mask.shape)
        mask = tf.reshape(mask, [-1, inputs.shape[2].value, inputs.shape[1].value]) # batch_size* hidden_size * sequence_length
        score_mask = tf.cast(mask, tf.bool)  # batch_size * sequence_length
        score_mask_values = 0 * tf.ones_like(inputs_trans)
        output = tf.where(score_mask, inputs_trans, score_mask_values)
        output = tf.transpose(output, [0,2,1])
        #output = tf.reduce_max(output, 1)
        output = tf.reduce_mean(output, 1)
        return output

class BiLstm(object):

        def __init__(self,args,data,ckpt_path): #seq_len,xvocab_size, label_size,ckpt_path,pos_size,type_size,data
                self.opt = args
                self.num_steps = data.max_sent_len
                self.topic_num_steps = data.max_topic_len
                self.num_class = 12
                self.num_chars = data.word_size
                self.ckpt_path=ckpt_path
                self.att_method = 'self_att' #'self_att' or 'topic_att' or 'att'
                self.opt_method = 'adam'
                self.loss_func = 'kl_loss' # 'kl_loss' or 'sigmoid_loss'
                self.batch_norm = False # True or False
                self.output_layer = 'FC' # 'FC' or 'SM'
                self.dropout_rate = 0.5 # [0.5,0.6,0.7]
                self.util= Util()
                sys.stdout.write('Building Graph ')
                self._build_model(args,embedding_matrix=data.pretrained)
                sys.stdout.write('graph built\n')

        def _build_model(self,flags,embedding_matrix):
                tf.reset_default_graph()
                tf.set_random_seed(123)
                self.input=tf.placeholder(shape=[None,self.num_steps], dtype=tf.int64)
                self.topic_input=tf.placeholder(shape=[None,self.topic_num_steps], dtype=tf.int64)
                self.weight = tf.placeholder(shape=[None, self.num_steps], dtype=tf.int64)
                self.topic_weight = tf.placeholder(shape=[None, self.topic_num_steps], dtype=tf.int64)
                self.length = tf.placeholder(shape=[None,], dtype=tf.int64)
                self.seqlength = tf.placeholder(shape=[None,], dtype=tf.int32)
                self.topic_length = tf.placeholder(shape=[None,], dtype=tf.int64)
                self.toplength = tf.placeholder(shape=[None,], dtype=tf.int32)
                if self.loss_func == 'sigmoid_loss':
                        self.label = tf.placeholder(shape=[None, self.num_class - 1], dtype=tf.float32)
                else:
                        self.target = tf.placeholder(shape=[None, self.num_class], dtype=tf.float32)
                #self.weight = [tf.placeholder(shape=[None, ], dtype=tf.float32, name='wi_{}'.format(t)) for t in range(self.num_steps)]
                self.keep_prob = tf.placeholder(tf.float32)  # drop out

                if embedding_matrix is not None:
                        self.embedding = tf.Variable(embedding_matrix, trainable=True, name="emb",dtype=tf.float32)#
                        #self.embedding_t = tf.Variable(embedding_matrix, trainable=True, name="emb_t",dtype=tf.float32)#
                else:
                        self.embedding = tf.get_variable("emb", [self.num_chars, self.emb_dim])
                        #self.embedding_t = tf.get_variable("emb_t", [self.num_chars, self.emb_dim])
                self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.input)
                self.inputs_emb_t = tf.nn.embedding_lookup(self.embedding, self.topic_input)

                # Bi-LSTM encoding for tweets
                with tf.variable_scope('LSTM_TW'):
                        cell = RNNCell.LSTMCell(num_units=flags.hidden_size, state_is_tuple=True)
                        #cell = RNNCell.GRUCell(num_units=flags.hidden_size)
                        dropout_cell = RNNCell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                        stacked_cell= RNNCell.MultiRNNCell([dropout_cell] * self.opt.num_layers, state_is_tuple=True)
                        #stacked_cell = RNNCell.MultiRNNCell([dropout_cell for _ in range(self.opt.num_layers)], state_is_tuple=True)
                        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked_cell,cell_bw=stacked_cell,dtype=tf.float32,
                                                                                                                          sequence_length=self.length,inputs=self.inputs_emb)
                        output_fw, output_bw = outputs
                        output= tf.concat([output_fw,output_bw], 2)
                        final_state = extract_axis_1(output, self.seqlength-1)

                # Bi-LSTM encoding for hashtags
                with tf.variable_scope('LSTM_HT'):
                        cell_t = RNNCell.LSTMCell(num_units=flags.hidden_size, state_is_tuple=True)
                        dropout_cell_t = RNNCell.DropoutWrapper(cell_t, output_keep_prob=self.keep_prob)
                        stacked_cell_t = RNNCell.MultiRNNCell([dropout_cell_t] * self.opt.num_layers, state_is_tuple=True)
                        outputs_t, states_t = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked_cell_t,cell_bw=stacked_cell_t,
                                                                                                                                  dtype=tf.float32,sequence_length=self.topic_length,
                                                                                                                                  inputs=self.inputs_emb_t)
                        output_fw_t, output_bw_t = outputs_t
                        output_t=tf.concat([output_fw_t,output_bw_t], 2)
                        #final_state_t = extract_axis_1(output_t, self.toplength - 1)
                        ###final_state_t = tf.reduce_max(output_t, 1)
                        final_state_t = max_pool_function(output_t, self.topic_weight)


                soft_dim = self.opt.hidden_size * 2
                #soft_dim_t = self.opt.hidden_size * 2
                # sentence representation
                ####output = tf.reduce_max(output, 1)
                #output, alphas1 = attention(output, soft_dim, return_alphas=True)
                print('attention_mechanism: '+self.att_method)
                if self.att_method == 'att':
                        output, self.alphas1 = attention_mask(output, soft_dim, self.weight, return_alphas=True)
                elif self.att_method == 'self_att':
                        output, self.alphas1 = self_attention_mask(output, soft_dim, self.weight, final_state, return_alphas=True)
                else:
                        output, self.alphas1 = self_topic_attention_mask(output, soft_dim, self.weight, final_state, final_state_t,
                                                                                                        return_alphas=True)
                output = tf.reshape(output, [-1, soft_dim])

                #with tf.name_scope("dropout-lstm-output"):
                        #output_drop = tf.nn.dropout(output, keep_prob=self.keep_prob)
                output_drop = output

                if self.output_layer == 'SM':
                        self.softmax_w = tf.get_variable("softmax_w", [soft_dim, self.num_class])
                        self.softmax_b = tf.get_variable("softmax_b", [self.num_class])
                        self.logits = tf.matmul(output_drop, self.softmax_w) + self.softmax_b
                else:
                        with tf.variable_scope("output-1-layer"):
                                hidden_layer = tf.contrib.layers.fully_connected(
                                        inputs=output_drop,
                                        num_outputs=self.opt.hidden_size,
                                        activation_fn=tf.nn.tanh,
                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                        weights_regularizer=None,
                                        biases_initializer=tf.constant_initializer(1e-04),
                                        scope="FC-1"
                                )
                        with tf.name_scope("dropout-output"):
                                h_drop = tf.nn.dropout(hidden_layer, keep_prob=self.keep_prob)
                        if self.loss_func == 'sigmoid_loss':
                                output_cls_num = self.num_class-1
                        else:
                                output_cls_num = self.num_class
                        with tf.variable_scope("output-2-layer"):
                                self.logits = tf.contrib.layers.fully_connected(
                                        inputs=h_drop,
                                        num_outputs=output_cls_num,
                                        activation_fn=None,
                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                        weights_regularizer=None,
                                        biases_initializer=tf.constant_initializer(1e-04),
                                        scope="FC-2"
                                )

                '''
                output_t, alphas1_t = attention_mask(output_t, soft_dim_t, self.topic_weight ,return_alphas=True)
                output_t = tf.reshape(output_t, [-1, soft_dim_t])

                final_output = tf.concat([output_t, output],1)

                # output layer for the aspect extraction
                self.softmax_w = tf.get_variable("softmax_w", [soft_dim+soft_dim_t, self.num_class])
                self.softmax_b = tf.get_variable("softmax_b", [self.num_class])
                self.logits = tf.matmul(final_output, self.softmax_w) + self.softmax_b
                '''

                if self.loss_func == 'sigmoid_loss':
                        self.decode_outputs_test = tf.nn.sigmoid(self.logits) #sigmoid: drop at least 3 points
                else:
                        self.decode_outputs_test = tf.nn.softmax(self.logits)

                #states_fw, states_bw = states
                if self.loss_func == 'sigmoid_loss':
                        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logits)
                else:
                        self.loss = kl_loss(self.target, self.decode_outputs_test) #sigmoid: drop at least 3 points

                if self.batch_norm:
                        self._opt = tf.train.AdamOptimizer(learning_rate=self.opt.learn_rate)
                        self._max_grad_norm = 1
                        grads_and_vars = self._opt.compute_gradients(self.loss)
                        grads_and_vars = [(g, v) if g is None else (tf.clip_by_norm(g, self._max_grad_norm), v) \
                                                          for g, v in grads_and_vars]
                        grads_and_vars = [(g, v) if g is None else (add_gradient_noise(g), v) \
                                                          for g, v in grads_and_vars]
                        nil_grads_and_vars = []
                        for g, v in grads_and_vars:
                                nil_grads_and_vars.append((g, v))
                        # if v.name in self._nil_vars:
                        ## nil_grads_and_vars.append((zero_nil_slot(g), v))
                        # pass
                        # else:
                        # nil_grads_and_vars.append((g, v))
                        self.train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")
                        self.embedding = tf.nn.l2_normalize(self.embedding, 1)
                else:
                        print('optimization_mechanism: ' + self.opt_method)
                        if self.opt_method == 'adagrad':
                                self._opt = tf.train.AdagradOptimizer(learning_rate=self.opt.learn_rate)  # bad
                        elif self.opt_method == 'adadelta':
                                self._opt = tf.train.AdadeltaOptimizer()  # too bad
                        else:
                                self.train_op = tf.train.AdamOptimizer(learning_rate=self.opt.learn_rate).minimize(self.loss)

                print("=" * 50)
                print("List of Variables:")
                for v in tf.trainable_variables():
                        print(v.name)
                print("=" * 50)

        '''Training and Evaluation'''
        def train(self, data, sess=None):
                saver = tf.train.Saver()
                if not sess:
                        sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1)) # create a session
                        #sess = tf.Session(tf.ConfigProto(device_count = {'GPU': 0}))  # create a session
                        sess.run(tf.global_variables_initializer())             # init all variables
                sys.stdout.write('\n Training started ...\n')
                best_f1=0.0
                best_epoch=0
                t1=time.time()
                for i in range(self.opt.epochs):
                        try:
                                loss, _, _ =self.run_epoch(sess,data,data.train,True)
                                #loss2, _ =self.run_epoch(sess,data,data.valid,True)
                                val_loss,predicts,_ = self.run_epoch(sess,data,data.valid,False)
                                t2=time.time()
                                print('epoch:%2d \t time:%.2f\tloss:%f\tvalid_loss:%f'%(i,t2-t1,loss,val_loss))
                                val_f1, val_f1_micro, val_acc = self.predict(data, sess)
                                t1=time.time()
                                if val_f1_micro > best_f1 :#if val_f1 > best_f1 and val_op_f1 > best_op_f1
                                        saver.save(sess, self.ckpt_path + self.opt.model_name + '.ckpt')
                                        best_f1=val_f1_micro
                                        best_epoch=i
                                sys.stdout.flush()
                        except KeyboardInterrupt:  # this will most definitely happen, so handle it
                                print('Interrupted by user at iteration {}'.format(i))
                                self.session = sess
                                return sess
                print('best valid f1:%f\tbest epoch:%d'%(best_f1,best_epoch))

        # prediction
        def predict(self, data, sess):
                label_num = self.num_class-1
                _, predicts,tgtswes = self.run_epoch(sess, data, data.test, False)
                gold_scores = data.test['label']
                y_true = np.array(gold_scores)
                if self.loss_func == 'sigmoid_loss':
                        threshold_list = [0.5]
                else:
                        threshold_list = [0.12] #0.09,0.1,0.11,0.12,0.13,0.05,0.09,0.15,0.2,0.25
                # threshold_list = [0.5] #sigmoid: drop at least 3 points
                acc_list = []
                macrof1_list = []
                microf1_list = []
                fout1 = open('true_label.txt', 'w')
                fout2 = open('pred_label.txt', 'w')
                fout3 = open('pred_prob.txt', 'w')
                fout4 = open('att_weights.txt', 'w')
                fin_lines = open('../data/SemEval18_test.txt', 'r', encoding = 'UTF-8').readlines()
                fout1.write('\n')
                fout2.write('\n')
                fout3.write('\n')
                fout4.write('\n')
                other_threshold_list = [1.0] #,0.3,0.1,0.5,0.7,0.9
                for threshold in threshold_list:
                        for other_thres in other_threshold_list:
                                pred_scores = []
                                for i in range(len(predicts)):
                                        inline = fin_lines[i+1]
                                        id = inline.split('\t')[0]
                                        fout1.write(id+'\t'+'tweet\t')
                                        fout2.write(id+'\t'+'tweet\t')
                                        fout3.write(id+'\t'+'tweet\t')
                                        pred_list = []
                                        pred = predicts[i]
                                        if pred[-1] > other_thres:
                                                for j in range(label_num):
                                                        pred_list.append(0)
                                                        fout1.write(str(y_true[i][j]) + '\t')
                                                        fout3.write(str(pred[j]) + '\t')
                                                        fout2.write('0\t')
                                        else:
                                                for j in range(label_num):
                                                        fout1.write(str(y_true[i][j])+ '\t')
                                                        fout3.write(str(pred[j])+ '\t')
                                                        if pred[j] < threshold:
                                                                pred_list.append(0)
                                                                fout2.write('0\t')
                                                        else:
                                                                pred_list.append(1)
                                                                fout2.write('1\t')
                                        fout1.write('\n')
                                        fout2.write('\n')
                                        fout3.write('\n')
                                        pred_scores.append(pred_list)
                                        for kk in range(tgtswes.shape[1]):
                                                fout4.write(str(tgtswes[i][kk])+'\t')
                                        fout4.write('\n')


                                y_pred = np.array(pred_scores)

                                acc = jaccard_similarity_score(y_true[:,:label_num], y_pred)

                                f1_micro = f1_score(y_true[:,:label_num], y_pred, average='micro')

                                f1_macro = f1_score(y_true[:,:label_num], y_pred, average='macro')

                                macrof1_list.append(f1_macro)
                                acc_list.append(acc)
                                microf1_list.append(f1_micro)

                                print('threshold:%f,other_thres:%f,acc:%f,f1_micro:%f, f1_macro:%f' % (threshold, other_thres,
                                                                                                                                                                           acc, f1_micro, f1_macro))
                f1_macro = max(macrof1_list)
                f1_micro = max(microf1_list)
                acc = max(acc_list)
                fout1.close()
                fout2.close()
                fout3.close()
                fout4.close()
                #os.system('python evaluate.py 3 pred_label.txt ../data/SemEval18_test.txt')

                if self.opt.use_ilp:
                        print("Now we are using ILP to add constraints")
                        ilp_predicts = self.ilp_solution(predicts)
                        ilp_fout1 = open('ilp_true_label.txt', 'w')
                        ilp_fout2 = open('ilp_pred_label.txt', 'w')
                        for i in range(len(predicts)):
                                ilp_fout1.write(str(i) + '\t' + 'tweet\t')
                                ilp_fout2.write(str(i) + '\t' + 'tweet\t')
                                pred = ilp_predicts[i]
                                for j in range(label_num):
                                        ilp_fout1.write(str(y_true[i][j]) + '\t')
                                        ilp_fout2.write(str(pred[j]) + '\t')
                                ilp_fout1.write('\n')
                                ilp_fout2.write('\n')
                        ilp_fout1.close()
                        ilp_fout2.close()
                        ilp_acc = jaccard_similarity_score(y_true[:, :label_num], ilp_predicts[:, :label_num])

                        ilp_f1_micro = f1_score(y_true[:, :label_num], ilp_predicts[:, :label_num], average='micro')

                        ilp_f1_macro = f1_score(y_true[:, :label_num], ilp_predicts[:, :label_num], average='macro')
                        print('after ilp acc:%f,f1_micro:%f, f1_macro:%f' % (ilp_acc, ilp_f1_micro, ilp_f1_macro))

                return f1_macro, f1_micro, acc

        def run_epoch(self, sess, data,data_type,is_train):
                losses = []
                num_batch = data.gen_batch_num(data_type)
                predicts=None
                tgtswes = None
                for i in range(num_batch):
                        input, target, label, weight, length,topic_input,topic_weight,topic_length =data.gen_batch(data_type, i)
                        #print(input)
                        if is_train:
                                feed_dict = self.get_feed(input,target,label, weight,length,topic_input,topic_weight,topic_length,
                                                                                  keep_prob=self.dropout_rate)
                                _, loss_v, predict = sess.run([self.train_op, self.loss, self.decode_outputs_test],
                                                                                                          feed_dict)
                        else:
                                feed_dict = self.get_feed(input,target,label, weight,length,topic_input,topic_weight,topic_length,
                                                                                  keep_prob=1.)
                                loss_v, predict, tgtsw= sess.run([self.loss, self.decode_outputs_test, self.alphas1], feed_dict)
                        losses.append(np.mean(loss_v))

                        if predicts is None:
                                predicts = predict
                                if not is_train:
                                    tgtswes = tgtsw
                        else:
                                predicts = np.concatenate((predicts, predict))
                                if not is_train:
                                    tgtswes = np.concatenate((tgtswes, tgtsw))

                return np.mean(losses),predicts, tgtswes

        def restore_last_session(self):
                saver = tf.train.Saver()
                sess = tf.Session()  # create a session
                saver.restore(sess, self.ckpt_path + self.opt.model_name + '.ckpt')
                print('model restored')
                return sess

        def get_feed(self, input, target, label, weight, length, topic_input,topic_weight,topic_length, keep_prob):
                feed_dict={self.input:input}
                #feed_dict.update({self.target[t]: target[t] for t in range(self.num_steps)})
                if self.loss_func == 'sigmoid_loss':
                        feed_dict[self.label] = label
                else:
                        feed_dict[self.target] = target
                feed_dict[self.weight] = weight
                #feed_dict.update({self.weight[t]: weight[t] for t in range(self.num_steps)})
                feed_dict[self.length]=length
                feed_dict[self.seqlength] = length
                feed_dict[self.topic_input]= topic_input
                feed_dict[self.topic_weight] = topic_weight
                feed_dict[self.topic_length]=topic_length
                feed_dict[self.toplength]=topic_length
                feed_dict[self.keep_prob] = keep_prob  # dropout prob
                return feed_dict
















