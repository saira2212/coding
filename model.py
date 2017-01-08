import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.python.ops import seq2seq

import numpy as np

class Model():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1
        if args.cell == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.cell == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.cell == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        def build_unirnn(args):
            cell = cell_fn(args.rnn_size, state_is_tuple=True)

            self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=True)
            self.cell2 = cell_dummy = cell_fn(args.rnn_size, state_is_tuple=True)

            self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
            self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
            self.initial_state = cell.zero_state(args.batch_size, tf.float32)
            self.initial_state2 = cell_dummy.zero_state(args.batch_size, tf.float32)

            with tf.variable_scope('rnnlm'):
                softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
                softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                    input_embeddings = tf.nn.embedding_lookup(embedding, self.input_data)
                    inputs = tf.unpack(input_embeddings, axis=1)
            def loop(prev, _):
                prev = tf.matmul(prev, softmax_w) + softmax_b
                prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
                return tf.nn.embedding_lookup(embedding, prev_symbol)

            outputs, self.last_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
            # dummy state in this case
            self.last_state2 = cell_dummy.zero_state(args.batch_size, tf.float32)
            output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
            self.logits = tf.matmul(output, softmax_w) + softmax_b

        def build_birnn(args):
            self.cell = fw_cell = cell_fn(args.rnn_size, state_is_tuple=True)
            self.cell2 = bw_cell = cell_fn(args.rnn_size, state_is_tuple=True)

            self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
            self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
            self.initial_state = fw_cell.zero_state(args.batch_size, tf.float32)
            self.initial_state2 = bw_cell.zero_state(args.batch_size, tf.float32)

            with tf.variable_scope('rnnlm'):
                softmax_w = tf.get_variable("softmax_w", [2*args.rnn_size, args.vocab_size])
                softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                    input_embeddings = tf.nn.embedding_lookup(embedding, self.input_data)
                    inputs = tf.unpack(input_embeddings, axis=1)

            outputs, self.last_state, self.last_state2 = rnn.bidirectional_rnn(fw_cell,
                                                                               bw_cell,
                                                                               inputs,
                                                                               initial_state_fw=self.initial_state,
                                                                               initial_state_bw=self.initial_state2,
                                                                               dtype=tf.float32)
            output = tf.reshape(tf.concat(1, outputs), [-1, 2*args.rnn_size])
            self.logits = tf.matmul(output, softmax_w) + softmax_b

        if args.model == 'uni':
            build_unirnn(args)
        else:
            build_birnn(args)

        self.probs = tf.nn.softmax(self.logits)
        self.loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
        self.cost = tf.reduce_sum(self.loss) / args.batch_size / args.seq_length
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def eval(self, sess, chars, vocab, text):
        batch_size = 200
        state = sess.run(self.cell.zero_state(1, tf.float32))
        state2 = sess.run(self.cell2.zero_state(1, tf.float32))
        x = [vocab[c] if c in vocab else vocab['UNK'] for c in text]
        x = [vocab['<S>']] + x + [vocab['</S>']]
        total_len = len(x) - 1
        # pad x so the batch_size divides it
        while len(x) % 200 != 1:
            x.append(vocab[' '])
        y = np.array(x[1:]).reshape((-1, batch_size))
        x = np.array(x[:-1]).reshape((-1, batch_size))

        total_loss = 0.0
        for i in range(x.shape[0]):
            feed = {self.input_data: x[i:i+1, :], self.targets: y[i:i+1, :],
                    self.initial_state: state, self.initial_state2: state2}
            [state, state2, loss] = sess.run([self.last_state, self.last_state2, self.loss], feed)
            total_loss += loss.sum()
        # need to subtract off loss from padding tokens
        total_loss -= loss[total_len % batch_size - batch_size:].sum()
        avg_entropy = total_loss / len(text)
        return np.exp(avg_entropy)  # this is the perplexity

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        state2 = sess.run(self.cell2.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state, self.initial_state2: state2}
            [state, state2] = sess.run([self.last_state, self.last_state2], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state, state2] = sess.run([self.probs, self.last_state, self.last_state2], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret


