import numpy as np
import operator

def pos(tag):
    onehot = 0
    if tag == 'NN' or tag == 'NNS':
        onehot = 1
    elif tag == 'FW':
        onehot = 2
    elif tag == 'NNP' or tag == 'NNPS':
        onehot = 3
    elif 'VB' in tag:
        onehot = 4
    else:
        onehot = 5

    return onehot


def chunk(tag):
    onehot = 0
    if 'NP' in tag:
        onehot = 1
    elif 'VP' in tag:
        onehot = 2
    elif 'PP' in tag:
        onehot = 3
    elif tag == 'O':
        onehot = 4
    else:
        onehot = 5

    return onehot


def capital(word):
    if ord(word[0]) >= 'A' and ord(word[0]) <= 'Z':
        return 1
    else:
        return 2


def get_input(FILE_NAME, MAX_DOCUMENT_LENGTH, word_to_idx):
    words = []
    pos1 = []
    pos2 = []
    cap = []
    tag = []

    sentence = []
    sentence_pos1 = []
    sentence_pos2 = []
    sentence_cap = []
    sentence_tag = []

    # get max words in sentence
    max_sentence_length = MAX_DOCUMENT_LENGTH  # findMaxLenght(FILE_NAME)
    sentence_length = 0

    print ("max sentence size is : " + str(max_sentence_length))

    for line in open(FILE_NAME):
        if line in ['\n', '\r\n']:
            # print("aa"+str(sentence_length) )
            for _ in range(max_sentence_length - sentence_length):
                tag.append(np.asarray([0, 0, 0, 0, 0]))
                temp = 0
                words.append(temp)
                pos1.append(temp)
                pos2.append(temp)
                cap.append(temp)
            sentence.append(words)
            sentence_pos1.append(pos1)
            sentence_pos2.append(pos2)
            sentence_cap.append(cap)
            sentence_tag.append(np.asarray(tag))

            sentence_length = 0
            words = []
            pos1 = []
            pos2 = []
            cap = []
            tag = []
        else:
            assert (len(line.split()) == 4)
            if sentence_length > max_sentence_length:
                continue
            sentence_length += 1
            temp = word_to_idx[line.split()[0]]
            temp_pos1 = pos(line.split()[1])  # adding pos embeddings
            temp_pos2 = chunk(line.split()[2])  # adding chunk embeddings
            temp_cap = capital(line.split()[0])  # adding capital embedding
            words.append(temp)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
            cap.append(temp_cap)

            t = line.split()[3]

            # Five classes 0-None,1-Person,2-Location,3-Organisation,4-Misc

            if t.endswith('O'):
                tag.append(np.asarray([1, 0, 0, 0, 0]))
            elif t.endswith('PER'):
                tag.append(np.asarray([0, 1, 0, 0, 0]))
            elif t.endswith('LOC'):
                tag.append(np.asarray([0, 0, 1, 0, 0]))
            elif t.endswith('ORG'):
                tag.append(np.asarray([0, 0, 0, 1, 0]))
            elif t.endswith('MISC'):
                tag.append(np.asarray([0, 0, 0, 0, 1]))
            else:
                print("error in input" + str(t))

    assert (len(sentence) == len(sentence_tag))
    return np.asarray(sentence), np.asarray(sentence_pos1), np.asarray(sentence_pos2), np.asarray(
        sentence_cap), sentence_tag


def get_vocab(file_list):
    max_sent_len = 0
    word_to_idx = {}

    # Starts at 2 for padding
    idx = 1
    sentence_length = 0
    words = []

    for filename in file_list:
        f = open(filename, "r")
        for line in f:
            if line in ['\n', '\r\n']:
                max_sent_len = max(max_sent_len, len(words))
                for word in words:
                    if not word in word_to_idx:
                        word_to_idx[word] = idx
                        idx += 1
                sentence_length = 0
                words = []
            else:
                assert (len(line.split()) == 4)
                sentence_length += 1
                temp = line.split()[0]
                words.append(temp)

        f.close()
    return max_sent_len, word_to_idx


def load_bin_vec(fname, vocab):
    """
	Loads 300x1 word vecs from Google (Mikolov) word2vec
	"""
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


if __name__ == "__main__":

    num_classes = 5
    BASE_DIR = "ner"  # path to coNLL data set
    f_names = [BASE_DIR + "/eng.train", BASE_DIR + "/eng.testa", BASE_DIR + "/eng.testb"]
    w2v_path = '/home/jfyu/torch/1.bin'

    max_sent_len, word_to_idx = get_vocab(f_names)

    print(max_sent_len)

    with open('conll_word_mapping.txt', 'w+') as embeddings_f:
        embeddings_f.write("*PADDING* 0\n")
        for word, idx in sorted(word_to_idx.items(), key=operator.itemgetter(1)):
            embeddings_f.write("%s %d\n" % (word, idx))

    w2v = load_bin_vec(w2v_path, word_to_idx)
    V = len(word_to_idx) + 1
    print 'Vocab size:', V

    # Not all words in word_to_idx are in w2v.
    # Word embeddings initialized to random Unif(-0.25, 0.25)
    embed = np.array(np.random.uniform(-0.25, 0.25, (V, len(w2v.values()[0]))), dtype=np.float32)
    embed[0] = 0
    for word, vec in w2v.items():
        embed[word_to_idx[word]] = vec

    print(embed.shape)

    V_pos1 = 6
    pos1_dim = 50
    embed_pos1 = np.array(np.random.uniform(-0.25, 0.25, (V_pos1, pos1_dim)), dtype=np.float32)
    embed_pos1[0] = 0

    V_pos2 = 6
    pos2_dim = 50
    embed_pos2 = np.array(np.random.uniform(-0.25, 0.25, (V_pos2, pos2_dim)), dtype=np.float32)
    embed_pos2[0] = 0

    V_cap = 3
    cap_dim = 10
    embed_cap = np.array(np.random.uniform(-0.25, 0.25, (V_cap, cap_dim)), dtype=np.float32)
    embed_cap[0] = 0

    train_word, train_pos1, train_pos2, train_cap, y_train = get_input(BASE_DIR + "/eng.train", max_sent_len,
                                                                       word_to_idx)
    test_word, test_pos1, test_pos2, test_cap, y_test = get_input(BASE_DIR + "/eng.testa", max_sent_len, word_to_idx)
    val_word, val_pos1, val_pos2, val_cap, y_val = get_input(BASE_DIR + "/eng.testb", max_sent_len, word_to_idx)

    y_train = np.asarray(y_train).astype(int)
    print(y_train.shape)

    y_test = np.asarray(y_test).astype(int)

    y_val = np.asarray(y_val).astype(int)

    np.save('./data/word.npy', embed)
    np.save('./data/pos1.npy', embed_pos1)
    np.save('./data/pos2.npy', embed_pos2)
    np.save('./data/cap.npy', embed_cap)

    np.save('./data/train_word.npy', train_word)
    np.save('./data/train_pos1.npy', train_pos1)
    np.save('./data/train_pos2.npy', train_pos2)
    np.save('./data/train_cap.npy', train_cap)
    np.save('./data/train_y.npy', y_train)

    np.save('./data/test_word.npy', test_word)
    np.save('./data/test_pos1.npy', test_pos1)
    np.save('./data/test_pos2.npy', test_pos2)
    np.save('./data/test_cap.npy', test_cap)
    np.save('./data/testall_y.npy', y_test)

    np.save('./data/val_word.npy', val_word)
    np.save('./data/val_pos1.npy', val_pos1)
    np.save('./data/val_pos2.npy', val_pos2)
    np.save('./data/val_cap.npy', val_cap)
    np.save('./data/val_y.npy', y_val)
