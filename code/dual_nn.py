"""
dual_nn.py

Dual Model with a vanilla feed-forward encoder. Has multiple heads for each of the three levels, as 
well as a head for level selection.
"""
import numpy as np
import tensorflow as tf

PAD, PAD_ID = "<<PAD>>", 0
UNK, UNK_ID = "<<UNK>>", 1

class NNDual():
    def __init__(self, l0_corpus, l1_corpus, l2_corpus, l0_commands, l1_commands, l2_commands,
                 embedding_size=30, h1_size=60, h2_size=50, epochs=10, batch_size=16):
        """
        Instantiates and Trains Model using the given set of parallel corpora.

        :param <lvl>_corpus: List of Tuples, where each tuple has two elements:
                    1) List of source sentence tokens
                    2) List of target sentence tokens
        :param <lvl>_commands: List of Lists, where each element is one of the possible commands (labels)
        """
        self.l0_pc, self.l1_pc, self.l2_pc = l0_corpus, l1_corpus, l2_corpus
        self.l0_commands, self.l0_labels = l0_commands, {" ".join(x): i for (i, x) in enumerate(l0_commands)}
        self.l1_commands, self.l1_labels = l1_commands, {" ".join(x): i for (i, x) in enumerate(l1_commands)}
        self.l2_commands, self.l2_labels = l2_commands, {" ".join(x): i for (i, x) in enumerate(l2_commands)}        
        
        self.epochs, self.bsz = epochs, batch_size
        self.embedding_sz, self.h1_sz, self.h2_sz = embedding_size, h1_size, h2_size
        self.init = tf.truncated_normal_initializer(stddev=0.5)

        # Build Level Dictionary
        self.lvl_dict = {'L0': (self.l0_commands, self.l0_pc, self.l0_labels), 
                         'L1': (self.l1_commands, self.l1_pc, self.l1_labels), 
                         'L2': (self.l2_commands, self.l2_pc, self.l2_labels)}

        # Build Vocabulary
        self.word2id, self.id2word = self.build_vocabulary()

        # Vectorize Parallel Corpus
        self.train_x, self.train_y = self.vectorize()

        # Setup Placeholders
        self.X = tf.placeholder(tf.float32, shape=[None, len(self.word2id)], name='NL_Command')
        self.L0_Y = tf.placeholder(tf.int32, shape=[None], name='L0_ML_Command')
        self.L1_Y = tf.placeholder(tf.int32, shape=[None], name='L1_ML_Command')
        self.L2_Y = tf.placeholder(tf.int32, shape=[None], name='L2_ML_Command')
        self.LVL_Y = tf.placeholder(tf.int32, shape=[None], name='Level_Label')
        self.keep_prob = tf.placeholder(tf.float32, name='Dropout_Prob')

        # Build Inference Graph
        self.l0_logits, self.l1_logits, self.l2_logits, self.lvl_logits = self.inference()
        self.l0_probs, self.l1_probs = tf.nn.softmax(self.l0_logits), tf.nn.softmax(self.l1_logits)
        self.l2_probs, self.lvl_probs = tf.nn.softmax(self.l2_logits), tf.nn.softmax(self.lvl_logits)

        # Build Loss Computations
        self.l0_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.l0_logits,
                                                                                     self.L0_Y))
        self.l1_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.l1_logits,
                                                                                     self.L1_Y))
        self.l2_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.l2_logits,
                                                                                     self.L2_Y))
        self.lvl_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.lvl_logits,
                                                                                      self.LVL_Y))

        # Build Training Operations
        self.opt = tf.train.AdamOptimizer()
        self.l0_train_op = self.opt.minimize(self.l0_loss + self.lvl_loss)
        self.l1_train_op = self.opt.minimize(self.l1_loss + self.lvl_loss)
        self.l2_train_op = self.opt.minimize(self.l2_loss + self.lvl_loss)

        # Initialize all variables
        self.session = tf.Session()
        self.saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())

    def build_vocabulary(self):
        """
        Builds the vocabulary from the parallel corpus, adding the UNK ID.

        :return: Tuple of Word2Id, Id2Word Dictionaries.
        """
        vocab = set()
        for lvl in self.lvl_dict:
            for n, _ in self.lvl_dict[lvl][1]:
                for word in n:
                    vocab.add(word)

        id2word = [PAD, UNK] + list(vocab)
        word2id = {id2word[i]: i for i in range(len(id2word))}
        return word2id, id2word

    def vectorize(self):
        """
        Step through the Parallel Corpus, and convert each sequence to vectors.
        """
        x, y = {}, {}
        for lvl in self.lvl_dict:
            x[lvl], y[lvl] = [], []
            for nl, ml in self.lvl_dict[lvl][1]:
                nvec, mlab = np.zeros((len(self.word2id)), dtype=np.int32), self.lvl_dict[lvl][2][" ".join(ml)]
                for w in nl:
                    nvec[self.word2id.get(w, UNK_ID)] += 1
                x[lvl].append(nvec)
                y[lvl].append(mlab)
        
        for lvl in self.lvl_dict:
            x[lvl] = np.array(x[lvl], dtype=np.int32)
            y[lvl] = np.array(y[lvl], dtype=np.int32)
        return x, y

    def inference(self):
        """
        Compile the LSTM Classifier, taking the input placeholder, generating the softmax
        distribution over all possible reward functions.
        """
        # Shared Embedding
        E = tf.get_variable("Embedding", shape=[len(self.word2id), self.embedding_sz],
                            dtype=tf.float32, initializer=self.init)
        embedding = tf.matmul(self.X, E)
        embedding = tf.nn.dropout(embedding, self.keep_prob)

        # Shared ReLU Layer
        H1_W = tf.get_variable("Hidden_W1", shape=[self.embedding_sz, self.h1_sz], dtype=tf.float32,
                               initializer=self.init)
        H1_B = tf.get_variable("Hidden_B1", shape=[self.h1_sz], dtype=tf.float32,
                               initializer=self.init)
        h1 = tf.nn.relu(tf.matmul(embedding, H1_W) + H1_B)
        
        # Level-Specific Layers
        outputs = {}
        for i in ['L0', 'L1', 'L2']:
            # ReLU Layer
            H2_W = tf.get_variable("Hidden_W_%s" % i, shape=[self.h1_sz, self.h2_sz], dtype=tf.float32,
                                   initializer=self.init)
            H2_B = tf.get_variable("Hidden_B_%s" % i, shape=[self.h2_sz], dtype=tf.float32,
                                   initializer=self.init)
            hidden = tf.nn.relu(tf.matmul(h1, H2_W) + H2_B)
            hidden = tf.nn.dropout(hidden, self.keep_prob)

            # Output Layer
            O_W = tf.get_variable("Output_W_%s" % i, shape=[self.h2_sz, len(self.lvl_dict[i][0])],
                                  dtype=tf.float32, initializer=self.init)
            O_B = tf.get_variable("Output_B_%s" % i, shape=[len(self.lvl_dict[i][0])], dtype=tf.float32,
                                  initializer=self.init)
            output = tf.matmul(hidden, O_W) + O_B
            outputs[i] = output
        
        # Level-Selection Hidden Layer
        HW_LVL = tf.get_variable("Hidden_W_LVL", shape=[self.h1_sz, 20], dtype=tf.float32, 
                                 initializer=self.init)
        HB_LVL = tf.get_variable("Hidden_B_LVL", shape=[20], dtype=tf.float32, initializer=self.init)
        hidden = tf.nn.relu(tf.matmul(h1, HW_LVL) + HB_LVL)

        # Level-Selection Output Layer
        OW_LVL = tf.get_variable("Output_W_LVL", shape=[20, 3], dtype=tf.float32, initializer=self.init)
        OB_LVL = tf.get_variable("Output_B_LVL", shape=[3], dtype=tf.float32, initializer=self.init)
        output = tf.matmul(hidden, OW_LVL) + OB_LVL

        return outputs['L0'], outputs['L1'], outputs['L2'], output

    def fit(self, chunk_size):
        """
        Train the model, with the specified batch size and number of epochs.
        """
        # Run through epochs
        for e in range(self.epochs):
            curr_loss, batches = 0.0, 0.0
            for start, end in zip(range(0, len(self.train_x['L0'][:chunk_size]) - self.bsz, self.bsz),
                                  range(self.bsz, len(self.train_x['L0'][:chunk_size]), self.bsz)):
                l0_loss, l1_loss, l2_loss = 0, 0, 0
                if end < len(self.train_x['L0']):
                    l0_loss, _ = self.session.run([self.l0_loss + self.lvl_loss, self.l0_train_op],
                                                feed_dict={self.X: self.train_x['L0'][start:end],
                                                            self.keep_prob: 0.5,
                                                            self.L0_Y: self.train_y['L0'][start:end],
                                                            self.LVL_Y: np.zeros([self.bsz]) + 0})
                if end < len(self.train_x['L1']):
                    l1_loss, _ = self.session.run([self.l1_loss + self.lvl_loss, self.l1_train_op],
                                                feed_dict={self.X: self.train_x['L1'][start:end],
                                                            self.keep_prob: 0.5,
                                                            self.L1_Y: self.train_y['L1'][start:end],
                                                            self.LVL_Y: np.zeros([self.bsz]) + 1})
                if end < len(self.train_x['L2']):
                    l2_loss, _ = self.session.run([self.l2_loss + self.lvl_loss, self.l2_train_op],
                                                feed_dict={self.X: self.train_x['L2'][start:end],
                                                            self.keep_prob: 0.5,
                                                            self.L2_Y: self.train_y['L2'][start:end],
                                                            self.LVL_Y: np.zeros([self.bsz]) + 2})
                curr_loss += l0_loss + l1_loss + l2_loss
                batches += 1
            print 'Epoch %s Average Loss:' % str(e), curr_loss / batches

    def score(self, nl_command):
        """
        Given a natural language command, return predicted output and score.

        :return: List of tokens representing predicted command, and score.
        """
        seq = np.zeros((len(self.word2id)), dtype=np.int32)
        for w in nl_command:
            seq[self.word2id.get(w, UNK_ID)] += 1
        
        lvl = self.session.run(self.lvl_probs, feed_dict={self.X: [seq], self.keep_prob: 1.0})
        [pred_level] = np.argmax(lvl, axis=1)

        if pred_level == 0:
            y = self.session.run(self.l0_probs, feed_dict={self.X: [seq], self.keep_prob: 1.0})
            [pred_command] = np.argmax(y, axis=1)
            return self.l0_commands[pred_command], y[0][pred_command], pred_level, lvl[0][pred_level]
        elif pred_level == 1:
            y = self.session.run(self.l1_probs, feed_dict={self.X: [seq], self.keep_prob: 1.0})
            [pred_command] = np.argmax(y, axis=1)
            return self.l1_commands[pred_command], y[0][pred_command], pred_level, lvl[0][pred_level]
        elif pred_level == 2:
            y = self.session.run(self.l2_probs, feed_dict={self.X: [seq], self.keep_prob: 1.0})
            [pred_command] = np.argmax(y, axis=1)
            return self.l2_commands[pred_command], y[0][pred_command], pred_level, lvl[0][pred_level]
        else:
            raise UnicodeError()
    