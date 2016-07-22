# char-rnn-tensorflow

# New changes in this repo:

- Save the best model (i.e. minimum training loss) so far in the 'best' subfolder
- Options for using gensim word2vec embedding

# Requirements
- [Tensorflow](http://www.tensorflow.org)
- python libraries:
    - [PyYAML](http://pyyaml.org/)
    - [CherryPy](http://www.cherrypy.org/)

Installation for python libraries:
```
pip install pyyaml cherrypy
```

# Changes to be added:

- Add a web server for sampling
- Temperature [Pull request #28](https://github.com/sherjilozair/char-rnn-tensorflow/pull/28)
- Dropouts [Pull request #35](https://github.com/sherjilozair/char-rnn-tensorflow/pull/35)



# Readme from the upstream repo:

Multi-layer Recurrent Neural Networks (LSTM, RNN) for character-level language models in Python using Tensorflow.

Inspired from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

# Requirements
- [Tensorflow](http://www.tensorflow.org)

# Basic Usage
To train with default parameters on the tinyshakespeare corpus, run `python train.py`.

To sample from a checkpointed model, `python sample.py`.
# Roadmap
- Add explanatory comments
- Expose more command-line arguments
- Compare accuracy and performance with char-rnn
