# Improving Multi-Label Emotion Classification (MLEC) via Sentiment Classication (SC)

(Here the labels for MLEC contain 11 emotions, which are respectively anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust;
and the labels for SC are respectively neutral, positive and negative)

This repository contains the following components:
[](https://github.sc-corp.net/Snapchat/jianfei-experiments/tree/master/SemEval2018_EC_github/)

- [Data Preprocessor for changing the data format of SC to be consistent with MLEC](https://github.sc-corp.net/Snapchat/jianfei-experiments/tree/master/SemEval2018_EC_github/ec_lstm_kl_att_topic_other+sentiment_dasts2/) at `ec_preprocess_4_model.py`
- [Data Preprocessor for creating the pickle files for both MLEC and SC](https://github.sc-corp.net/Snapchat/jianfei-experiments/tree/master/SemEval2018_EC_github/ec_lstm_kl_att_topic_other+sentiment_dasts2/) at `ec_data_build_kl_stdtok_py3_v2.py` (Note that this version v2 filters some infrequent (less than 2) words as UNK token, which slightly sacrifices the performance on the test data but may generalize better on unseen test data.)
- [Base Model without SC](https://github.sc-corp.net/Snapchat/jianfei-experiments/tree/master/SemEval2018_EC_github/ec_lstm_kl_att_topic_other/) at `ec_main.py`
- [Proposed Dual Attention Model with SC](https://github.sc-corp.net/Snapchat/jianfei-experiments/tree/master/SemEval2018_EC_github/ec_lstm_kl_att_topic_other+sentiment_dasts2/) at `ec_main.py`

###For Training Stage
### Steps to run the codes:
- Step1: python ec_preprocess_4_model.py (https://github.sc-corp.net/Snapchat/jianfei-experiments/tree/master/SemEval2018_EC_github/ec_lstm_kl_att_topic_other+sentiment_dasts2/)
- Step2: python ec_data_build_kl_stdtok_py3_v2.py (https://github.sc-corp.net/Snapchat/jianfei-experiments/tree/master/SemEval2018_EC_github/ec_lstm_kl_att_topic_other+sentiment_dasts2/)
- Step3: python ec_main.py (https://github.sc-corp.net/Snapchat/jianfei-experiments/tree/master/SemEval2018_EC_github/ec_lstm_kl_att_topic_other/)
- Step4: python ec_main.py (https://github.sc-corp.net/Snapchat/jianfei-experiments/tree/master/SemEval2018_EC_github/ec_lstm_kl_att_topic_other+sentiment_dasts2/)

### Data for MLEC and SC
- [Data for MLEC] (https://github.sc-corp.net/Snapchat/jianfei-experiments/tree/master/SemEval2018_EC_github/data/) train, dev and test files from SemEval 2018 Task 1C.
- [Data for SC] (https://github.sc-corp.net/Snapchat/jianfei-experiments/tree/master/SemEval2018_EC_github/data_twitter/) merge the train, dev and test files from SemEval 2016 Task 4A together to have the 'twitter-2016_all.txt' file.

Note that since the format of [Data for SC] is different from [Data for MLEC], we need to run this as shown above - [Data Preprocessor for changing the data format of SC to be consistent with MLEC].


### Requirements

- Python 3.6
- [TensorFlow](https://www.tensorflow.org)
- [Scikit-Learn](http://scikit-learn.org/stable/index.html)
- [Numpy](http://www.numpy.org/)

### Running Examples and Results

- [Log files] (https://github.sc-corp.net/Snapchat/jianfei-experiments/tree/master/SemEval2018_EC_github/log_files/) To show the runing procedure and results, I also attach the log files of running our preprocessing, base model and proposed model codes under this folder.

###For Test Stage
### Prediction
- [Only for prediction] python ec_main_predict.py (https://github.sc-corp.net/Snapchat/jianfei-experiments/tree/master/SemEval2018_EC_github/ec_lstm_kl_att_topic_other+sentiment_dasts2/) You can easily change the input sentences in line 183-line 189 to make predictions.
