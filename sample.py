import numpy as np
import tensorflow as tf

import argparse
import time
import os
import cPickle

from utils import TextLoader
from model import Model

class Sample(object):
    model = None

    def sample(self, args):
        if self.model is None:
            # Allow sample to be usable outside of main()
            with open(os.path.join(args.save_dir, 'config.pkl')) as f:
                saved_args = cPickle.load(f)
            with open(os.path.join(args.save_dir, 'chars_vocab.pkl')) as f:
                self.chars, self.vocab = cPickle.load(f)
            self.model = Model(saved_args, True)

            with tf.Session() as sess:
                tf.initialize_all_variables().run()
                saver = tf.train.Saver(tf.all_variables())
                ckpt = tf.train.get_checkpoint_state(args.save_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    return self.model.sample(sess, self.chars, self.vocab, args.n, args.prime)
        else:
            with tf.Session() as sess:
                tf.initialize_all_variables().run()
                saver = tf.train.Saver(tf.all_variables())
                ckpt = tf.train.get_checkpoint_state(args.save_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    return self.model.sample(sess, self.chars, self.vocab, args.n, args.prime)

        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                       help='number of characters to sample')
    parser.add_argument('--prime', type=str, default=' ',
                       help='prime text')
    args = parser.parse_args()
    value = Sample().sample(args)
    if value:
        print value

if __name__ == '__main__':
    main()
