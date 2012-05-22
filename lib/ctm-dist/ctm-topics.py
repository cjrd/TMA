#! /usr/bin/python

# usage: python topics.py <beta-file> <vocab-file> <num words>

import sys, numpy

def print_topics(beta_file, vocab_file,
                 nwords = 25, out = sys.stdout):

    # get the vocabulary

    vocab = file(vocab_file, 'r').readlines()
    vocab = map(lambda x: x.split()[0], vocab)

    indices = range(len(vocab))
    topic = numpy.array(map(float, file(beta_file, 'r').readlines()))

    nterms  = len(vocab)
    ntopics = len(topic)/nterms
    topic   = numpy.reshape(topic, [ntopics, nterms])
    for i in range(ntopics):
        out.write('\ntopic %03d\n' % i)
        indices.sort(lambda x,y: -cmp(topic[i,x], topic[i,y]))
        for j in range(nwords):
            out.write('     '+vocab[indices[j]]+'\n')


if (__name__ == '__main__'):
     beta_file = sys.argv[1]
     vocab_file = sys.argv[2]
     nwords = int(sys.argv[3])
     print_topics(beta_file, vocab_file, nwords)
