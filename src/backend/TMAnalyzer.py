from itertools import imap, combinations
from math import exp
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
    """"
    Base class for topic model analyzers, e.g. LDAAnalyzer
    """
    
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

    def general_kf_ppt(self, trainf_list, testf_list, test_wc, param, start, stop, step, train_params, test_params, _handle_ppt):
        """
        general perplexity testing framework used by several algorithms
        """
        # TODO much of this could be parallelized with the appropriate resources
        # TODO fix repititious use of settings files (perhaps fix the c code so we can just pass these params)
        if len(trainf_list) != len(testf_list):
            print 'Train and test lists must be the same length in kf_perplexity'
            return None

        if start < 0 or stop < 0 or step < 0:
            start = self.get_param(param)
            stop = start
            step = 1

        k = len(trainf_list)
        orig_outdir = self.get_param('outdir')

        orig_corpusfile = self.get_param('corpusfile')
        ppts = [[] for i in range(k)]

        for i in  xrange(k):
            for param_val in xrange(start, stop+1, step):
                # train the model
                train_out = os.path.join(orig_outdir, 'kf'+ str(k) + '_f' + str(i) + '_' + param + str(param_val))
                if not os.path.exists(train_out):
                    os.mkdir(train_out)
                train_params["corpusfile"] = trainf_list[i]
                train_params["outdir"] = train_out
                train_params[param] = param_val
                self.set_params(train_params)
                self.do_analysis()
                
                if self.params['alg'] == 'lda' or self.params['alg'] == 'ctm': # TODO: better way to do alg specific params?
                    test_params["infname"] = "inf" + str(i)
                elif self.params['alg'] == 'hdp':
                    test_params["saved_model"] = os.path.join(train_out,'mode.bin')

                test_params["corpusfile"] = testf_list[i]
                self.set_params(test_params)
                self.do_analysis()
                ppts[i].append([param_val, _handle_ppt(self, test_wc[i])]) # TODO add error catching in case there was an inference problem
        self.set_params({'outdir':orig_outdir, 'corpusfile':orig_corpusfile}) # return the analyzer to its original state
        return ppts


    def get_params(self):
        """
        return all parameters stored in self.params
        """
        return self.params

    def get_param(self, param):
        """
        return 'param' from self.params
        @param: param desired 'param' from self.params
        @return: the parameter 'param' from self.params
        """
        return self.params[param]
        
    def set_params(self, dict):
        """
        Set the parameters in self.param to the input dictionary
        NOTE: prints a warning if parameter was not previously in dictionary
        @param: the dictionary of param/value pairs
        """
        for prm in dict.keys():
            if self.params.has_key(prm):
                self.params[prm] = dict[prm]
            else:
                self.params[prm] = dict[prm]
                print "WARNING: unkown parameter value: %s\nNOTE: first initialize each analyzer" % prm

    def init_rel_db(self):
        """
        Initialize the relationship (TMA) database by creating the appropriate tables
        """
        self.dbase = db(self.params['outdir'] + '/tma.sqlite')
        self.dbase.add_table("doc_doc (id INTEGER PRIMARY KEY, doc_a INTEGER, doc_b INTEGER, score FLOAT)")
        self.dbase.add_table("doc_topic (id INTEGER PRIMARY KEY, doc INTEGER, topic INTEGER, score FLOAT)")
        self.dbase.add_table("topics (id INTEGER PRIMARY KEY, title VARCHAR(100), score FLOAT)")
        self.dbase.add_table("topic_term (id INTEGER PRIMARY KEY, topic INTEGER, term INTEGER, score FLOAT)")
        self.dbase.add_table("topic_topic (id INTEGER PRIMARY KEY, topic_a INTEGER, topic_b INTEGER, score FLOAT)")
        self.dbase.add_table("doc_term (id INTEGER PRIMARY KEY, doc INTEGER, term INTEGER, score FLOAT)")
        self.dbase.add_table("terms (id INTEGER PRIMARY KEY, title VARCHAR(100), count INTEGER)")
        self.dbase.add_table("docs (id INTEGER PRIMARY KEY, title VARCHAR(100))")

    def create_db_indices(self):
        """
        Create indexes in TMA database for fast lookup
        """
        self.dbase.add_index('topics_score_idx ON topics(score)')
        self.dbase.add_index('terms_ct_idx ON terms(count)')
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
        
    def write_terms_table(self, terms_file=None, wcs={}):
        """
        Write the terms to the database (id, title)
        @param terms_file: the file containing 1 term per line where line 0 is id=0, line 1 is id=1, etc
        @param wcs: word count dictionary
        """
        self._check_tlist(terms_file)
        self.dbase.executemany('INSERT INTO terms(id, title, count) VALUES(?, ?, ?)', self._term_table_gen(wcs))

    def _term_table_gen(self, wcs):
        """
        called after tlist is initialized
        """
        for i, trm in enumerate(self.terms_list):
            if wcs.has_key(i):
                count = wcs[i]
            else:
                count = -1
            yield (i, trm, count)

            
    def write_docs_table(self, docs_file=None):
        """
        Write the documents to the database (id, title)
        @param docs_file: the file containing 1 title per line where line 0 is id=0, line 1 is id=1, etc
        """
        if not docs_file:
            docs_file = self.params['titlesfile']
        res = file_generator(open(docs_file, 'r'))
        self.dbase.executemany('INSERT INTO docs (id, title) VALUES(?, ?)',
                        ([i,j] for i,j in enumerate(imap(buffer, res)))) # each should be a list
    
    def write_topic_topic(self, top_term_mat):
        """
        Write the topic x topic matrix to the database
        @param top_term_mat: topics x terms matrix, should represent log-lieklihood for accurate calculations
        """
        # TODO make distance metric a user option
        execution_str = 'INSERT INTO topic_topic (id, topic_a, topic_b, score) VALUES(NULL, ?, ?, ?)'
        for i in xrange(top_term_mat.shape[0]):
            scores = 1/hellinger_distance(top_term_mat[i,:]**0.5, top_term_mat[i+1:,:]**0.5)
            scores[np.where(np.isinf(scores))] = -1
            res = generic_generator((i,)*len(scores), range(i+1, i+1+len(scores)), scores)
            self.dbase.executemany(execution_str, res)

    def write_topics_table(self, top_term_mat, doc_top_mat, indices=None, terms_file=None):
        """
        For each topic, write the first 3 most probably words to the database
        @param top_term_mat: topics x terms matrix, should represent log-lieklihood for accurate calculations
        @param indices the ordered indices of the topic-term matrix (will compute in function if not supplied)
        @param terms_file: the file of terms (default is self.params['vocabfile'])
        """
        self._check_tlist(terms_file)
        if indices is None:
            indices = self._get_rev_srt_ind(top_term_mat)
        title_list = []
        tscores = doc_top_mat.sum(0) # topic scores are the total log-prob across the docs
        for i in xrange(indices.shape[0]):
            title = "{%s, %s, %s}" % (self.terms_list[indices[i, 0]],
                                    self.terms_list[indices[i, 1]],
                                    self.terms_list[indices[i, 2]])
            title_list.append([i, title, tscores[i]])
        self.dbase.executemany('INSERT INTO topics (id, title, score) VALUES(?, ?, ?)', title_list)

    def write_topic_terms(self, top_term_mat):
        """
        Write the topic x term matrix to the database
        @param top_term_mat: topics x terms matrix, should represent log-lieklihood for accurate calculations
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
        Save the topic x term matrix to a python object for later comparison
        Note: comparing the terms pairwise is O(N^2) in the number of terms and incredibly slow/memory-consuming given > 5k terms
        therefore we compute the term-term scores dynamically
        @param top_term_mat: topics x terms matrix, does not need to be normalised but a larger score should be better
        """
        out_obj_file = os.path.join(self.params['outdir'],'top_term_mat.obj')
        pickle.dump(np.sqrt(top_term_mat.T),open(out_obj_file, 'wb'))
        
    def _get_rev_srt_ind(self, mat):
        """
        find the reverse sorted indices of the supplied matrix
        @param mat: the matrix that will be used to find the reverse sorted indices
        @return: the indices of mat sorted in descending order
        """
        return np.argsort(mat)[:,::-1] # this is ~ 4x faster than fliplr


    def write_doc_term(self, wordcount_file=None, return_wcs=True):
        """
        write the document-term relationship to the database, term-doc simply uses tf counts
        @param wordcount_file: a file in lda-c format with each line representing one document and
        term-id:term-count
        """
        # TODO should these be normalized?
        if wordcount_file is None:
            wordcount_file = self.params['corpusfile']

        if return_wcs:
            self.wcs = {}

        execution_str = 'INSERT INTO doc_term (id, doc, term, score) VALUES(NULL, ?, ?, ?)'
        self.dbase.executemany(execution_str, self._doc_term_gen(wordcount_file, return_wcs))
            
        if return_wcs:
            return self.wcs

    def _doc_term_gen(self, wordcount_file, return_wcs):
        for doc_no, doc in enumerate(open(wordcount_file, 'r')):
            doc = doc.split()[1:]
            for term in doc:
                trm_id = int(term.split(':')[0])
                trm_ct = int(term.split(':')[1])
                if return_wcs:
                    if self.wcs.has_key(trm_id):
                        self.wcs[trm_id] += trm_ct
                    else:
                        self.wcs[trm_id] = trm_ct
                yield (doc_no, trm_id, trm_ct)

    def write_doc_topic(self, doc_top_mat):
        """
        Write the doc-topic relationships to the db
        @param doc_top_mat: a document x topic matrix where higher scores indicate greater similarity
        """
        def _doc_topic_gen(doc_top_mat):
            for doc_no in xrange(doc_top_mat.shape[0]):
                doc = doc_top_mat[doc_no,:]
                for top_no in xrange(len(doc)):
                    yield (doc_no, top_no, doc[top_no])

        execution_str = 'INSERT INTO doc_topic (id, doc, topic, score) VALUES(NULL, ?, ?, ?)'
        self.dbase.executemany(execution_str, _doc_topic_gen(doc_top_mat))

    


            
    def write_doc_doc(self, doc_top_mat):
        """
        Write the doc-doc relationships to the db
        @param doc_top_mat: a document x topic matrix where higher scores indicate greater similarity
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
        """
        Load the terms list if the analyzer does not have the terms_list already loaded
        @param terms_file: the file with 1 term per line (line number is term id starting at 0)
        """
        if not hasattr(self, 'terms_list'):
            if not terms_file:
                terms_file = self.params['vocabfile']
            self.terms_list = []
            with open(terms_file) as tfile:
                for trm in tfile:
                    self.terms_list.append(trm.strip())
    


    