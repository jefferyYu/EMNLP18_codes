# Improving Multi-Label Emotion Classification (MLEC) via Sentiment Classication (SC)

(Here the labels for MLEC contain 11 emotions, which are respectively anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust;
and the labels for SC are respectively neutral, positive and negative)

This repository contains the following components:
[](../SemEval2018_EC_github/)

- [Data Preprocessor for changing the data format of SC to be consistent with MLEC] at `ec_preprocess_4_model.py`
- [Data Preprocessor for creating the pickle files for both MLEC and SC] at `ec_data_build_kl_stdtok_py3.py`
- [Base Model without SC] at `ec_main.py`
- [Proposed Dual Attention Model with SC] at `ec_main.py`


### Steps to run the codes:
- python ec_preprocess_4_model.py (../SemEval2018_EC_github/ec_lstm_kl_att_topic_other+sentiment_dasts2/)
- python ec_data_build_kl_stdtok_py3.py (../SemEval2018_EC_github/ec_lstm_kl_att_topic_other+sentiment_dasts2/)
- python ec_main.py (../SemEval2018_EC_github/ec_lstm_kl_att_topic_other/)
- python ec_main.py (../SemEval2018_EC_github/ec_lstm_kl_att_topic_other+sentiment_dasts2/)

### Data for MLEC and SC
- [Data for MLEC] (../SemEval2018_EC_github/data/) train, dev and test files from SemEval 2018 Task 1C.
- [Data for SC] (../SemEval2018_EC_github/data_twitter/) merge the train, dev and test files from SemEval 2016 Task 4A together to have the 'twitter-2016_all.txt' file.

Note that since the format of [Data for SC] is different from [Data for MLEC], we need to run this as shown above - [Data Preprocessor for changing the data format of SC to be consistent with MLEC].


### Requirements

- Python 3.x
- [TensorFlow](https://www.tensorflow.org)
- [Scikit-Learn](http://scikit-learn.org/stable/index.html)
- [Numpy](http://www.numpy.org/)

### Running Examples and Results

- [Log files] (../SemEval2018_EC_github/log_files/) To show the runing procedure and results, I also attach the log files of running our preprocessing, base model and proposed model codes under this folder.

