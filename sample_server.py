import os
import threading
import cherrypy
import argparse

from six.moves import cPickle
from model import Model

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default='8080',
                       help='port the server runs on')
    parser.add_argument('--environment', type=str, default='development',
                       help="environment the server runs in, e.g. 'development', 'production'")
    parser.add_argument('--model_dir', type=int, default='save',
                       help='directory to restore checkpointed models')
    args = parser.parse_args()
    server_config = {'server.socket_port': args.port, 'environment': args.environement}
    cherrypy.config.update(server_config)
    cherrypy.quickstart(SampleServer(args.model_dir))


class SampleServer(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.lock = threading.Lock()
        self.threaded_models = {}
        self._load()

    def load(self):
        with open(os.path.join(self.model_dir, 'chars_vocab.pkl'), 'rb') as f:
            self.chars, self.vocab = cPickle.load(f)
        with open(os.path.join(self.model_dir, 'config.pkl'), 'rb') as f:
            self.saved_args = cPickle.load(f)

    def _get_thread_model(self):
        tid = threading.get_ident()
        if tid in self.threaded_models:
            return self.threaded_models[tid]
        # else
        self.lock.acquire()
        if not tid in self.threaded_models:
            self.threaded_models[tid] = Model(self.saved_args, True)
        self.lock.release()
        return self.threaded_models[tid]

    @cherrypy.expose
    def index(self, prime='The ', n=200, sample_mode=2):
        model = self._get_thread_model()
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            saver = tf.train.Saver(tf.all_variables())
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Loaded {}".format(ckpt.model_checkpoint_path))
                result = model.sample(sess, self.chars, self.vocab, n, prime, sample_mode)
                return result


if __name__ == '__main__':
    main()