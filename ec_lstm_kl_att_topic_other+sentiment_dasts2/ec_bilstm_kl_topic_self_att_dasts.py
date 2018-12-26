import tensorflow as tf
import numpy as np
import sys
import time
from ec_util import Util
import tensorflow.contrib.rnn as RNNCell
from sklearn.metrics import f1_score, jaccard_similarity_score
#from ilp import Ilp
import os

def test_accuracy(pred, true):
    if pred.shape[0] != true.shape[0]:
        print("error: the length of two lists are not the same")
        return 0
    else:
        count = 0
        for i in range(pred.shape[0]):
            if pred[i] == true[i]:
                count += 1
        return float(count)/len(pred)

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

def self_spec_attention_mask(inputs, hidden_size, mask, sent_rep, prev_alpha, return_alphas=False):
        #inputs matrix: batch_size, sequence length, hidden_size
        # prev_alpha: batch_size * sequence_length
        # sent_rep: batch_size, hidden_size
        sequence_length = inputs.shape[1].value # the length of sequences processed in the antecedent RNN layer
        #hidden_size = inputs.shape[2].value # hidden size of the RNN layer
        #sent_reps = tf.tile(sent_rep, [1,sequence_length])
        sent_reps = tf.tile(sent_rep, [1,sequence_length]) # batch_size, sequence_len*hidden_size
        print(sent_reps.shape)
        #top_reps = tf.tile(top_rep, [1,sequence_length])

        # Attention mechanism - one
        W_omega = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.1))
        W_s = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.1))
        W_t = tf.Variable(tf.random_normal([1, hidden_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
        v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) +
                                tf.matmul(tf.reshape(sent_reps, [-1, hidden_size]), W_s) +
                                tf.matmul(tf.reshape(prev_alpha, [-1, 1]), W_t) + tf.reshape(b_omega, [1, -1]))

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

def fn(a):
        return a

def fn_zero(a):
        return 0 * a

class BiLstm(object):

        def __init__(self,args,data,ckpt_path): #seq_len,xvocab_size, label_size,ckpt_path,pos_size,type_size,data
                self.opt = args
                self.num_steps = data.max_sent_len
                self.topic_num_steps = data.max_topic_len
                self.num_class = 12
                self.sent_num_class = 3
                self.num_chars = data.word_size
                self.ckpt_path=ckpt_path
                self.att_method = 'self_att' #'self_att' or 'topic_att' or 'att'
                self.opt_method = 'adam'
                self.loss_func = 'kl_loss' # 'kl_loss' or 'sigmoid_loss'
                self.batch_norm = False # True or False
                self.output_layer = 'FC' # 'FC' or 'SM'
                self.dropout_rate = 0.5 # [0.5,0.6,0.7]
                self.lamb = 0.05
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
                self.istrain = tf.placeholder(tf.bool, shape=())
                if self.loss_func == 'sigmoid_loss':
                        self.label = tf.placeholder(shape=[None, self.num_class - 1], dtype=tf.float32)
                        self.sent_label = tf.placeholder(shape=[None,], dtype=tf.int32)
                else:
                        self.target = tf.placeholder(shape=[None, self.num_class], dtype=tf.float32)
                        self.sent_label = tf.placeholder(shape=[None,], dtype=tf.int32)
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

                soft_dim = self.opt.hidden_size * 2

                # Bi-LSTM encoding for tweets
                with tf.variable_scope('LSTM_Tgt'):
                        tgt_cell = RNNCell.LSTMCell(num_units=flags.hidden_size, state_is_tuple=True)
                        #cell = RNNCell.GRUCell(num_units=flags.hidden_size)
                        tgt_dropout_cell = RNNCell.DropoutWrapper(tgt_cell, output_keep_prob=self.keep_prob)
                        tgt_stacked_cell= RNNCell.MultiRNNCell([tgt_dropout_cell] * self.opt.num_layers, state_is_tuple=True)
                        #stacked_cell = RNNCell.MultiRNNCell([dropout_cell for _ in range(self.opt.num_layers)], state_is_tuple=True)
                        tgt_outputs, tgt_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=tgt_stacked_cell,cell_bw=tgt_stacked_cell,
                                                                                                                          dtype=tf.float32, sequence_length=self.length,
                                                                                                                          inputs=self.inputs_emb)
                        tgt_output_fw, tgt_output_bw = tgt_outputs
                        tgt_output=     tf.concat([tgt_output_fw,tgt_output_bw], 2)
                        tgt_final_state = extract_axis_1(tgt_output, self.seqlength-1)

                        #soft_dim_t = self.opt.hidden_size * 2
                        # sentence representation
                        ####output = tf.reduce_max(output, 1)
                        #output, alphas1 = attention(output, soft_dim, return_alphas=True)
                        print('attention_mechanism: '+self.att_method)
                        print('lambda: '+str(self.lamb))
                        if self.att_method == 'att':
                                tgt_output, self.tgt_alphas1 = attention_mask(tgt_output, soft_dim, self.weight, return_alphas=True)
                        elif self.att_method == 'self_att':
                                tgt_output, self.tgt_alphas1 = self_attention_mask(tgt_output, soft_dim, self.weight, tgt_final_state,
                                                                                                          return_alphas=True)
                                #tgt_output, self.tgt_alphas1 = self_spec_attention_mask(tgt_output, soft_dim, self.weight, tgt_final_state,
                                                                                                                                                #self.alphas1, return_alphas=True)
                        tgt_output = tf.reshape(tgt_output, [-1, soft_dim])

                with tf.variable_scope('LSTM_Shared'):
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

                        #soft_dim_t = self.opt.hidden_size * 2
                        # sentence representation
                        ####output = tf.reduce_max(output, 1)
                        #output, alphas1 = attention(output, soft_dim, return_alphas=True)
                        real_alpha = tf.cond(self.istrain,lambda:fn_zero(self.tgt_alphas1), lambda:fn(self.tgt_alphas1))
                        print('attention_mechanism: '+self.att_method)
                        if self.att_method == 'att':
                                output, self.alphas1 = attention_mask(output, soft_dim, self.weight, return_alphas=True)
                        elif self.att_method == 'self_att':
                                #output, self.alphas1 = self_attention_mask(output, soft_dim, self.weight, final_state,
                                                                                                                   #return_alphas=True)
                                output, self.alphas1 = self_spec_attention_mask(output, soft_dim, self.weight, final_state,real_alpha,
                                                                                                                   return_alphas=True)

                        output = tf.reshape(output, [-1, soft_dim])
                #with tf.name_scope("dropout-lstm-output"):
                        #output_drop = tf.nn.dropout(output, keep_prob=self.keep_prob)
                tgt_output_drop = tf.concat([output,tgt_output], 1)
                src_output_drop = output

                if self.output_layer == 'SM':
                        self.softmax_w = tf.get_variable("softmax_w", [soft_dim*2, self.num_class])
                        self.softmax_b = tf.get_variable("softmax_b", [self.num_class])
                        self.logits = tf.matmul(tgt_output_drop, self.softmax_w) + self.softmax_b
                else:
                        with tf.variable_scope("output-1-layer"):
                                hidden_layer = tf.contrib.layers.fully_connected(
                                        inputs=tgt_output_drop,
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

                self.softmax_sentw = tf.get_variable("softmax_sentw", [soft_dim, self.sent_num_class])
                self.softmax_sentb = tf.get_variable("softmax_sentb", [self.sent_num_class])
                self.sent_logits = tf.matmul(src_output_drop, self.softmax_sentw) + self.softmax_sentb
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
                self.decode_outputs_sent_test = tf.nn.softmax(self.sent_logits)

                #states_fw, states_bw = states
                if self.loss_func == 'sigmoid_loss':
                        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logits)
                else:
                        self.loss = kl_loss(self.target, self.decode_outputs_test)+ \
                                                self.lamb*tf.losses.cosine_distance(self.tgt_alphas1, self.alphas1, dim=1)
                self.sent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.sent_label,
                                                                                                                                                logits=self.sent_logits)

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
                                self.train_sent_op = tf.train.AdamOptimizer(learning_rate=self.opt.learn_rate).minimize(self.sent_loss)

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
                                loss, sent_loss, _, _, best_f1 =self.run_epoch(sess,data,data.train,data.train2, saver, best_f1)
                                #loss2, _ =self.run_epoch(sess,data,data.valid,True)
                                val_loss, val_sent_loss, predicts,sent_predicts,tgtswes, tgtwes, srcswes = self.run_test_epoch(
                                        sess,data,data.valid,data.valid2)
                                t2=time.time()
                                print('epoch:%2d \t time:%.2f\tloss:%f\tvalid_loss:%f\tsentloss:%f\tsentvalid_loss:%f'%
                                          (i,t2-t1,loss,val_loss,sent_loss, val_sent_loss))
                                val_f1, val_f1_micro, val_acc = self.predict(data, sess)
                                t1=time.time()
                                if val_acc > best_f1 :#if val_f1 > best_f1 and val_op_f1 > best_op_f1
                                        saver.save(sess, self.ckpt_path + self.opt.model_name + '.ckpt')
                                        best_f1=val_acc
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
                _, _, predicts, sent_predicts,tgtswes, tgtwes, srcswes = self.run_test_epoch(
                        sess, data, data.test, data.test2)
                gold_scores = data.test['label']
                sent_gold_scores = data.test2['target']
                model_pred = np.argmax(sent_predicts, axis=1)
                acc = test_accuracy(model_pred, sent_gold_scores)
                print('sentiment classification:'+str(acc))
                y_true = np.array(gold_scores)
                if self.loss_func == 'sigmoid_loss':
                        threshold_list = [0.5]
                else:
                        threshold_list = [0.12] #0.09,0.1,0.11,0.12,0.13
                # threshold_list = [0.5] #sigmoid: drop at least 3 points
                acc_list = []
                macrof1_list = []
                microf1_list = []
                fout1 = open('true_label.txt', 'w')
                fout2 = open('pred_label.txt', 'w')
                fout3 = open('pred_prob.txt', 'w')
                fout4 = open('tgt_shared_att_weights.txt', 'w')
                fout5 = open('tgt_spec_att_weights.txt', 'w')
                fout6 = open('src_shared_att_weights.txt', 'w')
                fin_lines = open('../data/SemEval18_test.txt', 'r', encoding='UTF-8').readlines()
                fout1.write('\n')
                fout2.write('\n')
                fout3.write('\n')
                fout4.write('\n')
                fout5.write('\n')
                fout6.write('\n')
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
                                                fout5.write(str(tgtwes[i][kk])+'\t')
                                                fout6.write(str(srcswes[i][kk])+'\t')
                                        fout4.write('\n')
                                        fout5.write('\n')
                                        fout6.write('\n')

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
                fout5.close()
                fout6.close()
                #os.system('python evaluate.py 3 pred_label.txt ../data/SemEval18_test.txt')

                return f1_macro, f1_micro, acc
        
        def only_predict(self, sess, input,weight,length,topic_input,topic_weight,topic_length):
                label_num = self.num_class-1
                all_predicts, all_sent_predicts = self.run_pred_epoch(
                                sess, input,weight,length,topic_input,topic_weight,topic_length)
                predicts = all_predicts[0]
                sent_predicts = all_sent_predicts[0]
                model_pred = np.argmax(sent_predicts, axis=1) # neutral 0 positive 1 negative 2
                print('now write sentiment classification result')
                fout1 = open('pred_sent_label.txt', 'w')
                fout1.write(' \tneutral:0\tpositive:1\tnegative:2\n')
                for i in range(len(sent_predicts)):
                        fout1.write(str(i)+'\t'+str(model_pred[i])+'\n')
                fout1.close()
        
                print('now write emotion classification result')
                if self.loss_func == 'sigmoid_loss':
                        threshold_list = [0.5]
                else:
                        threshold_list = [0.12] #0.09,0.1,0.11,0.12,0.13
                # threshold_list = [0.5] #sigmoid: drop at least 3 points
                acc_list = []
                macrof1_list = []
                microf1_list = []
                fout2 = open('pred_emotion_label.txt', 'w')
                #fout3 = open('pred_prob.txt', 'w')
                fout2.write(' \tanger\tanticipation\tdisgust\tfear\tjoy\tlove\toptimism\tpessimism\tsadness\tsurprise\ttrust\n')
                #fout3.write('\n')
                other_threshold_list = [1.0] #,0.3,0.1,0.5,0.7,0.9
                for threshold in threshold_list:
                        for other_thres in other_threshold_list:
                                pred_scores = []
                                for i in range(len(predicts)):
                                        fout2.write(str(i)+'\t')
                                        #fout3.write('id'+'\t'+'tweet\t')
                                        pred_list = []
                                        pred = predicts[i]
                                        if pred[-1] > other_thres:
                                                for j in range(label_num):
                                                        pred_list.append(0)
                                                        #fout3.write(str(pred[j]) + '\t')
                                                        fout2.write('0\t')
                                        else:
                                                for j in range(label_num):
                                                        #fout3.write(str(pred[j])+ '\t')
                                                        if pred[j] < threshold:
                                                                pred_list.append(0)
                                                                fout2.write('0\t')
                                                        else:
                                                                pred_list.append(1)
                                                                fout2.write('1\t')
                                        fout2.write('\n')
                                        #fout3.write('\n')
                                        pred_scores.append(pred_list)
                fout2.close()
                
        def joint_predict(self, sess, input,weight,length,topic_input,topic_weight,topic_length):
                label_num = self.num_class-1
                all_predicts, all_sent_predicts = self.run_pred_epoch(
                                sess, input,weight,length,topic_input,topic_weight,topic_length)
                
                sent_pred_probs = all_sent_predicts[0]
                sent_preds = np.argmax(sent_pred_probs, axis=1) # neutral 0 positive 1 negative 2


                predicts = all_predicts[0]
                threshold = 0.12 #0.09,0.1,0.11,0.12,0.13               
                pred_scores = []
                pred_prob_scores = []
                for i in range(len(predicts)):
                        pred_list = []
                        pred_prob_list = []
                        pred = predicts[i]
                        for j in range(label_num):
                                p = pred[j]
                                if p < threshold:
                                        pred_list.append(0)
                                        pred_prob_list.append(0.5 + 0.5*float(p-threshold)/threshold)
                                else:
                                        pred_list.append(1)
                                        pred_prob_list.append(0.5 + 0.5*float(p-threshold)/(1-threshold))
                        pred_scores.append(pred_list)
                        pred_prob_scores.append(pred_prob_list)
                return sent_preds, sent_pred_probs, pred_scores, pred_prob_scores

        def run_epoch(self, sess, data,data_type,sent_data_type, saver, best_f1):
                losses = []
                sent_losses = []
                num_batch = data.gen_batch_num(data_type)
                sent_num_batch = data.gen_batch_num(sent_data_type)
                predicts=None
                sent_predicts=None
                j = 0

                for i in range(sent_num_batch):
                        if j == num_batch:
                                j = 0
                                print('##################################################################')
                        if j % 50 == 0:
                                val_loss, val_sent_loss, predicts,sent_predicts,_, _,_,= \
                                        self.run_test_epoch(sess,data, data.valid,data.valid2)
                                print('batch:%f\tvalid_loss:%f\tsentvalid_loss:%f'%(j,val_loss, val_sent_loss))
                                val_f1, val_f1_micro, val_acc = self.predict(data, sess)
                                if val_acc > best_f1 :#if val_f1 > best_f1 and val_op_f1 > best_op_f1
                                        saver.save(sess, self.ckpt_path + self.opt.model_name + '.ckpt')
                                        best_f1=val_acc
                                sys.stdout.flush()
                        input, target, label, weight, length,topic_input,topic_weight,topic_length =data.gen_batch(data_type, j)
                        j += 1
                        #print(target[:5])
                        input2, target2, weight2, length2,topic_input2,topic_weight2,topic_length2 =\
                                data.gen_sent_batch(sent_data_type, i)
                        #print(target2[:5])
                        #print(input)
                        feed_dict = self.get_feed(input,target,label, weight,length,topic_input,topic_weight,topic_length,
                                                                          keep_prob=self.dropout_rate)
                        feed_dict2 = self.get_sent_feed(input2,target2,weight2,length2,topic_input2,topic_weight2,topic_length2,
                                                                          keep_prob=self.dropout_rate)
                        _, loss_v, predict = sess.run([self.train_op, self.loss, self.decode_outputs_test],
                                                                                                  feed_dict)
                        _, sent_loss_v, sent_predict = sess.run([self.train_sent_op, self.sent_loss,
                                                                                                         self.decode_outputs_sent_test], feed_dict2)
                        losses.append(np.mean(loss_v))
                        sent_losses.append(np.mean(sent_loss_v))

                        if predicts is None:
                                predicts = predict
                        else:
                                predicts = np.concatenate((predicts, predict))
                        if sent_predicts is None:
                                sent_predicts = sent_predict
                        else:
                                sent_predicts = np.concatenate((sent_predicts, sent_predict))

                return np.mean(losses),np.mean(sent_losses),predicts, sent_predicts, best_f1

        def run_test_epoch(self, sess, data,data_type,sent_data_type):
                losses = []
                sent_losses = []
                num_batch = data.gen_batch_num(data_type)
                sent_num_batch = data.gen_batch_num(sent_data_type)
                predicts=None
                sent_predicts=None
                tgtswes=None
                tgtwes=None
                srcswes=None

                for i in range(num_batch):
                        input, target, label, weight, length,topic_input,topic_weight,topic_length =data.gen_batch(data_type, i)
                        #print(target[:5])
                        input2, target2, weight2, length2,topic_input2,topic_weight2,topic_length2 =\
                                data.gen_sent_batch(sent_data_type, i)

                        feed_dict = self.get_feed(input,target,label, weight,length,topic_input,topic_weight,topic_length,
                                                                                  keep_prob=1.)
                        feed_dict2 = self.get_sent_feed(input2,target2,weight2,length2,topic_input2,topic_weight2,topic_length2,
                                                                                  keep_prob=1.)
                        loss_v, predict, tgtsw, tgtw= sess.run([self.loss, self.decode_outputs_test,
                                                                                                  self.alphas1, self.tgt_alphas1], feed_dict)
                        sent_loss_v, sent_predict, srcsw = sess.run([self.sent_loss, self.decode_outputs_sent_test,
                                                                                                   self.alphas1], feed_dict2)
                        losses.append(np.mean(loss_v))
                        sent_losses.append(np.mean(sent_loss_v))

                        if predicts is None:
                                predicts = predict
                                tgtswes = tgtsw
                                tgtwes = tgtw
                        else:
                                predicts = np.concatenate((predicts, predict))
                                tgtswes = np.concatenate((tgtswes, tgtsw))
                                tgtwes = np.concatenate((tgtwes, tgtw))
                        if sent_predicts is None:
                                sent_predicts = sent_predict
                                srcswes = srcsw
                        else:
                                sent_predicts = np.concatenate((sent_predicts, sent_predict))
                                srcswes = np.concatenate((srcswes, srcsw))

                return np.mean(losses),np.mean(sent_losses),predicts, sent_predicts, tgtswes, tgtwes, srcswes
                
        def run_pred_epoch(self, sess, input,weight,length,topic_input,topic_weight,topic_length):
                predicts=None
                sent_predicts=None
                
                feed_dict = self.get_pred_feed(input,weight,length,topic_input,topic_weight,topic_length,
                                                                          keep_prob=1.)
                feed_dict2 = self.get_pred_sent_feed(input,weight,length,topic_input,topic_weight,topic_length,
                                                                          keep_prob=1.)
                predict= sess.run([self.decode_outputs_test], feed_dict)
                sent_predict = sess.run([self.decode_outputs_sent_test], feed_dict2)

                if predicts is None:
                        predicts = predict
                else:
                        predicts = np.concatenate((predicts, predict))
                if sent_predicts is None:
                        sent_predicts = sent_predict
                else:
                        sent_predicts = np.concatenate((sent_predicts, sent_predict))

                return predicts, sent_predicts

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
                feed_dict[self.istrain] = False
                #feed_dict.update({self.weight[t]: weight[t] for t in range(self.num_steps)})
                feed_dict[self.length]=length
                feed_dict[self.seqlength] = length
                feed_dict[self.topic_input]= topic_input
                feed_dict[self.topic_weight] = topic_weight
                feed_dict[self.topic_length]=topic_length
                feed_dict[self.toplength]=topic_length
                feed_dict[self.keep_prob] = keep_prob  # dropout prob
                return feed_dict

        def get_sent_feed(self, input, label, weight, length, topic_input,topic_weight,topic_length, keep_prob):
                feed_dict={self.input:input}
                #feed_dict.update({self.target[t]: target[t] for t in range(self.num_steps)})
                feed_dict[self.sent_label] = label
                feed_dict[self.weight] = weight
                feed_dict[self.istrain] = True
                #feed_dict.update({self.weight[t]: weight[t] for t in range(self.num_steps)})
                feed_dict[self.length]=length
                feed_dict[self.seqlength] = length
                feed_dict[self.topic_input]= topic_input
                feed_dict[self.topic_weight] = topic_weight
                feed_dict[self.topic_length]=topic_length
                feed_dict[self.toplength]=topic_length
                feed_dict[self.keep_prob] = keep_prob  # dropout prob
                return feed_dict
                
        def get_pred_feed(self, input, weight, length, topic_input,topic_weight,topic_length, keep_prob):
                feed_dict={self.input:input}
                feed_dict[self.weight] = weight
                feed_dict[self.istrain] = False # for judging sentiment or emotion tasks
                feed_dict[self.length]=length
                feed_dict[self.seqlength] = length
                feed_dict[self.topic_input]= topic_input
                feed_dict[self.topic_weight] = topic_weight
                feed_dict[self.topic_length]=topic_length
                feed_dict[self.toplength]=topic_length
                feed_dict[self.keep_prob] = keep_prob  # dropout prob
                return feed_dict

        def get_pred_sent_feed(self, input, weight, length, topic_input,topic_weight,topic_length, keep_prob):
                feed_dict={self.input:input}
                feed_dict[self.weight] = weight
                feed_dict[self.istrain] = True # for judging sentiment or emotion tasks
                feed_dict[self.length]=length
                feed_dict[self.seqlength] = length
                feed_dict[self.topic_input]= topic_input
                feed_dict[self.topic_weight] = topic_weight
                feed_dict[self.topic_length]=topic_length
                feed_dict[self.toplength]=topic_length
                feed_dict[self.keep_prob] = keep_prob  # dropout prob
                return feed_dict
        

















