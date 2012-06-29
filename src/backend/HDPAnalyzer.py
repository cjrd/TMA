import pdb
from tmanalyzer import TMAnalyzer
import os        
import math     
from time import time
import numpy as np

class HDPAnalyzer(TMAnalyzer):
    def __init__(self, params):
        """                                          
              hdp [options]
              general parameters:
              --algorithm:      train or test, not optional
              --data:           data file, in lda-c format, not optional
              --directory (outdir): save directory, not optional
              --max_iter:       the max number of iterations, default 1000
              --save_lag:       the saving lag, default 100 (-1 means no savings for intermediate results)
              --random_seed:    the random seed, default from the current time
              --init_topics:    the initial number of topics, default 0

              training parameters:
              --gamma_a:        shape for 1st-level concentration parameter, default 1.0
              --gamma_b:        scale for 1st-level concentration parameter, default 1.0
              --alpha_a:        shape for 2nd-level concentration parameter, default 1.0
              --alpha_b:        scale for 2nd-level concentration parameter, default 1.0
              --sample_hyper:   sample 1st and 2nd-level concentration parameter, yes or no, default "no"
              --eta:            topic Dirichlet parameter, default 0.5
              --split_merge:    try split-merge or not, yes or no, default "no"
              --restrict_scan:  number of intermediate scans, default 5 (-1 means no scan)

              testing parameters:
              --saved_model:    path for saved model, not optional 
        """                               
                                                               
        # replace default parameters with passed in parameters
        hdpparams = {
            'algorithm':'train',
            'corpusfile':'corpus.data',
            'outdir':'hdpout',
            'max_iter':1000,
            'save_lag':-1,
            'init_topics':0,
            'gamma_a':1.0,
            'gamma_b':1.0,
            'alpha_a':1.0,
            'alpha_b':1.0,
            'sample_hyper':'no',
            'eta':0.5,
            'split_merge':'no',
            'restrict_scan': 5,
            'vocabfile':'vocab.txt',
            'titlesfile':'titles.txt',
            'hdpdir':'hdp',
            'ndocs':-1,
            'wordct':-1,
            'saved_model':'mode.bin',
            'timelimit':-1
            }

        for prm in params.keys():
            if hdpparams.has_key(prm):
                hdpparams[prm] = params[prm]
            else:
                raise Exception("unkown parameter value for HDPAnalyzer: %s" % prm)
        hdpparams['alg'] = 'hdp'       
        super(HDPAnalyzer, self).__init__(hdpparams)
    
    def do_analysis(self):
        """
        Execute the HDP with the specified parameters
        """
        if self.params['algorithm'] == 'train':
            cmd = 'ulimit -t %(timelimit)d; %(hdpdir)s/hdp --data %(corpusfile)s --algorithm %(algorithm)s --directory %(outdir)s\
               --max_iter %(max_iter)d --save_lag %(save_lag)d --init_topics %(init_topics)i --gamma_a %(gamma_a)f\
               --gamma_b %(gamma_b)f --alpha_a %(alpha_a)f --alpha_b %(alpha_b)f --sample_hyper %(sample_hyper)s\
               --eta %(eta)f --split_merge %(split_merge)s --restrict_scan %(restrict_scan)s' % self.params
        elif self.params['algorithm'] == 'testlike':
           cmd = 'ulimit -t %(timelimit)d; %(hdpdir)s/hdp --algorithm %(algorithm)s --data %(corpusfile)s\
            --saved_model %(saved_model)s  --directory %(outdir)s' % self.params
        print cmd
        stime = time()
        os.system(cmd)
        print 'finished HDP analysis in %f seconds' % (time()-stime)

    def kf_perplexity(self, trainf_list, testf_list, test_wc, param='ntopics', start = -1, stop = -1, step = 5):
        """
        Calculates the perplexity given the training and testing files in trainf_list and testf_list, respectively
        @param trainf_list: list of paths of the training corpora
        @param testf_list: list of paths of the corresponding testing corpora (must be same length as trainf_list so that each training corpus has a testing corpus)
        @param test_wc: the total number of words in each test corpus
        @return the perplexity for each fold
        """

        train_params = {'algorithm':'train'}
        test_params = {'algorithm':'testlike'}

        def _handle_ppt(self, test_wci):
            lhood = float(open(os.path.join(self.params['outdir'], 'test-loglike.dat')).readline().strip())
            return round((-1*lhood/test_wci), 3)
        return self.general_kf_ppt(trainf_list, testf_list, test_wc, param, start, stop, step, train_params, test_params, _handle_ppt)

    def createJSLikeData(self):
        """
        transform the model likelihood data for plot display on analysis page
        """
        linfile = open('%s/state.log' % self.params['outdir'], 'r')          
        ldata = linfile.readlines()   
        linfile.close()  
        jsout = open('%s/js_likelihood.dat'% self.params['outdir'],'w')   
        jsout.write('[');                            
        for i, line in enumerate(ldata[1:]):   
            lik = line.strip().split()[4]
            jsout.write(lik)
            if not i == len(ldata)-2:
                jsout.write(',')
            else:
                jsout.write(']')
        jsout.close()

    def create_relations(self):
        """
        This method uses the document termcounts, topics x terms matrix and documents x topics matrix to determine the following relationships:
        - term x term
        - topic x term
        - topic x topic
        - document x document
        - document x topic
        - document x term

        NOTE: this method should be called after 'do_analysis'
        """
        self.init_rel_db()

        # doc-term (STD)
        wc_db = self.write_doc_term()

        # write the vocab to the database (STD)
        self.write_terms_table(wcs=wc_db)

        # write doc title to database (STD)
        self.write_docs_table()
        top_term_mat = np.loadtxt('%s/mode-topics.dat' % self.params['outdir'])
        top_term_mat /= top_term_mat.sum(1)[:,np.newaxis] # normalize
        top_term_mat = np.log(top_term_mat)

        # topic_terms -- expects log probabilities
        self.write_topic_terms(top_term_mat)

        # topic_topic -- expects probabilities
        self.write_topic_topic(np.exp(top_term_mat))

        # term_term  -- expects probabilities
        self.write_term_term(np.exp(top_term_mat))

        # load/form top_doc matrix
        mw_mat = np.loadtxt('%s/mode-word-assignments.dat' % self.params['outdir'], skiprows=1)
        ndocs = np.max(mw_mat[:,0]) + 1
        ntops = np.max(mw_mat[:,2]) + 1

        doc_top_mat = np.zeros([ndocs, ntops])
        for i in xrange(mw_mat.shape[0]):
            doc_top_mat[mw_mat[i, 0], mw_mat[i, 2]] += 1

        # doc_doc
        doc_top_mat /= doc_top_mat.sum(1)[:, np.newaxis]
        self.write_doc_doc(doc_top_mat**0.5)

        # doc_topic
        self.write_doc_topic(doc_top_mat)

        # write topics, i.e. top 3 terms (STD)
        self.write_topics_table(top_term_mat=top_term_mat, doc_top_mat=doc_top_mat)

        # create indices for fast lookup
        self.create_db_indices()
            
    
