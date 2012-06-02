from itertools import imap, combinations
from src.backend.db import db
from src.backend.generate_db import generate_db
import pdb
import os    
from time import time
from src.backend.math_utils import hellinger_distance
from src.backend.utils import file_generator, generic_generator
import numpy as np
import cPickle as pickle

class TMAnalyzer(object):
    """"Base class for topic model analyzers, e.g. LDAAnalyzer"""
    
    def __init__(self, params): 
        if not params.has_key('outdir'):
            params['outdir'] = '.' # the outdir is required
        if not os.path.exists(params['outdir']):
            os.makedirs(params['outdir'])   
        self.params = params
        
    def do_analysis(self):
        raise NotImplementedError( "Should have implemented do_analysis in subclass analyzer" )
        
    
    def create_relations(self):
        raise NotImplementedError( "Should have implemented do_analysis in subclass analyzer" )
    
    def kf_perplexity(self, trainf_list, testf_list, test_wc, param='ntopics', start = -1, stop = -1, step = 5):
        print "Perplexity calculation not implemented, will return None"
        return None

    def get_params(self):
        return self.params

    def get_param(self, param):
        return self.params[param]
        
    def set_params(self, dict):
        for prm in dict.keys():
            if self.params.has_key(prm):
                self.params[prm] = dict[prm]
            else:
                self.params[prm] = dict[prm]
                print "WARNING: unkown parameter value: %s\nNOTE: first initialize each analyzer" % prm

    def init_rel_db(self):
        self.dbase = db(self.params['outdir'] + '/tma.sqlite')
        self.dbase.add_table("doc_doc (id INTEGER PRIMARY KEY, doc_a INTEGER, doc_b INTEGER, score FLOAT)")
        self.dbase.add_table("doc_topic (id INTEGER PRIMARY KEY, doc INTEGER, topic INTEGER, score FLOAT)")
        self.dbase.add_table("topics (id INTEGER PRIMARY KEY, title VARCHAR(100))")
        self.dbase.add_table("topic_term (id INTEGER PRIMARY KEY, topic INTEGER, term INTEGER, score FLOAT)")
        self.dbase.add_table("topic_topic (id INTEGER PRIMARY KEY, topic_a INTEGER, topic_b INTEGER, score FLOAT)")
        self.dbase.add_table("doc_term (id INTEGER PRIMARY KEY, doc INTEGER, term INTEGER, score FLOAT)")
        self.dbase.add_table("terms (id INTEGER PRIMARY KEY, title VARCHAR(100))")
        self.dbase.add_table("docs (id INTEGER PRIMARY KEY, title VARCHAR(100))")

    def create_db_indices(self):
        self.dbase.add_index('doc_doc_idx1 ON doc_doc(doc_a)')
        self.dbase.add_index('doc_doc_idx2 ON doc_doc(doc_b)')
        self.dbase.add_index('doc_doc_idx_score ON doc_doc(score)')
        self.dbase.add_index('doc_topic_idx1 ON doc_topic(doc)')
        self.dbase.add_index('doc_topic_idx2 ON doc_topic(topic)')
        self.dbase.add_index('doc_topic_idx_score ON doc_topic(score)')
        self.dbase.add_index('topic_term_idx1 ON topic_term(topic)')
        self.dbase.add_index('topic_term_idx2 ON topic_term(term)')
        self.dbase.add_index('topic_term_score ON topic_term(score)')
        self.dbase.add_index('topic_topic_idx1 ON topic_topic(topic_a)')
        self.dbase.add_index('topic_topic_idx2 ON topic_topic(topic_b)')
        self.dbase.add_index('topic_topic_score ON topic_topic(score)')
        self.dbase.add_index('doc_term_idx1 ON doc_term(doc)')
        self.dbase.add_index('doc_term_idx2 ON doc_term(term)')
        self.dbase.add_index('doc_term_score ON doc_term(term)') 
        
    def write_terms_table(self, terms_file=None):
        self._check_tlist(terms_file)
        for i, trm in enumerate(self.terms_list):
            self.dbase.execute('INSERT INTO terms(id, title) VALUES(?, ?)', (i, trm)) # explictly write id so term ids match (not off by one)

    def write_docs_table(self, docs_file=None):
        if not docs_file:
            docs_file = self.params['titlesfile']
        res = file_generator(open(docs_file, 'r'))
        self.dbase.executemany('INSERT INTO docs (id, title) VALUES(?, ?)',
                        ([i,j] for i,j in enumerate(imap(buffer, res)))) # each should be a list
    
    def write_topic_topic(self, top_term_mat):
        """
        Write the topic x topic matrix to the database
        @param top_term_mat: topics x terms matrix
        """
        # TODO make distance metric a user option
        execution_str = 'INSERT INTO topic_topic (id, topic_a, topic_b, score) VALUES(NULL, ?, ?, ?)'
        for i in xrange(top_term_mat.shape[0]):
            scores = 1/hellinger_distance(top_term_mat[i,:]**0.5, top_term_mat[i+1:,:]**0.5)
            scores[np.where(np.isinf(scores))] = -1
            res = generic_generator((i,)*len(scores), range(i+1, i+1+len(scores)), scores)
            self.dbase.executemany(execution_str, res)

    def write_topic_terms(self, top_term_mat):
        """
        Write the topic x term matrix to the database
        @param top_term_mat: topics x terms matrix, does not need to be normalised but a larger score should be better
        """
        ntops = top_term_mat.shape[0]
        nterms = top_term_mat.shape[1]
        execution_str = 'INSERT INTO topic_term (id, topic, term, score) VALUES(NULL, ?, ?, ?)'
        for topic_no in xrange(ntops):
            topic = top_term_mat[topic_no,:]
            res = generic_generator((topic_no,)*nterms, range(nterms), topic)
            self.dbase.executemany(execution_str, res)

        
    def write_term_term(self, top_term_mat):
        """

        """
        out_obj_file = os.path.join(self.params['outdir'],'top_term_mat.obj')
        pickle.dump(np.sqrt(top_term_mat.T),open(out_obj_file, 'wb'))
        
    def write_topics_table(self, top_term_mat, indices=None, terms_file=None):
        """
        For each topic, write the first 3 most probably words to the database
        @param top_term_mat: topics x terms matrix
        @param indices the ordered indices of the topic-term matrix (will compute in function if not supplied)
        @param terms_file: the file of terms (default is self.params['vocabfile'])
        """
        self._check_tlist(terms_file)
        if indices is None:
            indices = self._get_rev_srt_ind(top_term_mat)
        title_list = []
        for i in xrange(indices.shape[0]):
            title = "{%s, %s, %s}" % (self.terms_list[indices[i, 0]],
                                    self.terms_list[indices[i, 1]],
                                    self.terms_list[indices[i, 2]])
            title_list.append([i, title])
        self.dbase.executemany('INSERT INTO topics (id, title) VALUES(?, ?)', title_list)

    def _get_rev_srt_ind(self, mat):
        """
        find the reverse sorted indices of the supplied matrix
        @param mat: the matrix that will be used to find the reverse sorted indices
        """
        return np.argsort(mat)[:,::-1] # this is ~ 4x faster than fliplr


    def write_doc_term(self, wordcount_file=None):
        """

        """
        if wordcount_file is not None:
            wordcount_file = self.params['corpusfile']

        execution_str = 'INSERT INTO doc_term (id, doc, term, score) VALUES(NULL, ?, ?, ?)'
        for doc_no, doc in enumerate(open(wordcount_file, 'r')):
            doc = doc.split()[1:]
            terms = {}
            for term in doc:
                terms[int(term.split(':')[0])] = int(term.split(':')[1])

            keys = terms.keys()
            res = generic_generator((doc_no,)*len(keys),
                                    keys, (terms[i] for i in keys))
            self.dbase.executemany(execution_str, res)

    def write_doc_topic(self, doc_top_mat):
        """
        """
        execution_str = 'INSERT INTO doc_topic (id, doc, topic, score) VALUES(NULL, ?, ?, ?)'
        for doc_no in xrange(doc_top_mat.shape[0]):
            doc = doc_top_mat[doc_no,:]
            res = generic_generator((doc_no,)*len(doc), range(len(doc)), doc)
            self.dbase.executemany(execution_str, res)
        
    def write_doc_doc(self, doc_top_mat):
        """
        """
        ndocs = doc_top_mat.shape[0]
        scores = np.zeros([ndocs, ndocs])
        for i in xrange(ndocs):
            scores[i, i+1:] = 1/hellinger_distance(doc_top_mat[i,:]**0.5, doc_top_mat[i+1:,:]**0.5)
        scores[np.where(np.isinf(scores))] = -1
        scores = scores + scores.T # for accurate top K doc-docs
        score_inds = self._get_rev_srt_ind((scores))[:,:30] # take the top thirty related docs (lower bound) TODO make an option?
        db_list = []
        idxs = {} # so we don't have duplicates in the database
        for i in xrange(scores.shape[0]):
            for j in score_inds[i,:]:
                j = int(j)
                minv = min(i,j)
                maxv = max(i,j)
                if not idxs.has_key('%i %i' % (minv,maxv)):
                        db_list.append((minv, maxv, round(scores[minv,maxv], 3))) # TODO this could probably be replaced with a generator
                        idxs['%i %i' % (minv,maxv)] = 1
        self.dbase.executemany("INSERT INTO  doc_doc('id', 'doc_a', 'doc_b', 'score') VALUES(NULL, ?, ?, ?)", db_list)
        
        

    def _check_tlist(self, terms_file):
        if not hasattr(self, 'terms_list'):
            if not terms_file:
                terms_file = self.params['vocabfile']
            self.terms_list = []
            with open(terms_file) as tfile:
                for trm in tfile:
                    self.terms_list.append(trm.strip())
    


    