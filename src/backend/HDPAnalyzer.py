import pdb
from TMAnalyzer import TMAnalyzer
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
            'saved_model':'mode.bin'
            }

        for prm in params.keys():
            if hdpparams.has_key(prm):
                hdpparams[prm] = params[prm]
            else:
                raise Exception("unkown parameter value for HDPAnalyzer: %s" % prm)
        hdpparams['alg'] = 'hdp'       
        super(HDPAnalyzer, self).__init__(hdpparams)
    
    def do_analysis(self):

         # do the analysis
         if self.params['algorithm'] == 'train':
             cmd = '%(hdpdir)s/hdp --data %(corpusfile)s --algorithm %(algorithm)s --directory %(outdir)s\
                --max_iter %(max_iter)d --save_lag %(save_lag)d --init_topics %(init_topics)i --gamma_a %(gamma_a)f\
                --gamma_b %(gamma_b)f --alpha_a %(alpha_a)f --alpha_b %(alpha_b)f --sample_hyper %(sample_hyper)s\
                --eta %(eta)f --split_merge %(split_merge)s --restrict_scan %(restrict_scan)s' % self.params
         elif self.params['algorithm'] == 'testlike':
            cmd = '%(hdpdir)s/hdp --algorithm %(algorithm)s --data %(corpusfile)s\
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

        # TODO fix code repetition with other analyzers
        
        if len(trainf_list) != len(testf_list):
            print 'Train and test lists must be the same length in LDAAnalyzer kf_perplexity'
            return None

        if start < 0 or stop < 0 or step < 0:
            start = self.get_param(param)
            stop = start
            step = 1

        k = len(trainf_list)
        orig_outdir = self.get_param('outdir')

        orig_corpusfile = self.get_param('corpusfile')
        train_params = {'algorithm':'train'}
        test_params = {'algorithm':'testlike'}
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
                test_params["saved_model"] = os.path.join(train_out,'mode.bin')
               # train_params["outdir"] = os.path.join(orig_outdir, "inf" + str(i)) # technically this isn't a dir since it also prefixes the file
                test_params["corpusfile"] = testf_list[i]
                self.set_params(test_params)
                self.do_analysis()
                lhood = float(open(os.path.join(self.params['outdir'], 'test-loglike.dat')).readline().strip())
                ppts[i].append([param_val, round(math.exp(-1*lhood/test_wc[i]), 3)]) # TODO add error catching in case there was an inference problem

        self.set_params({'outdir':orig_outdir, 'corpusfile':orig_corpusfile}) # return the analyzer to its original state
        return ppts

    def createJSLikeData(self):
        # transform the likelihood data
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
        NOTE: this method should be called after 'do_analysis'
        """
        self.init_rel_db()

        # write the vocab to the database (STD)
        self.write_terms_table()

        # write doc title to database (STD)
        self.write_docs_table()
        top_term_mat = np.loadtxt('%s/mode-topics.dat' % self.params['outdir'])
        top_term_mat /= top_term_mat.sum(1)[:,np.newaxis] # normalize

        # write topics, i.e. top 3 terms (STD) -- need
        self.write_topics_table(top_term_mat=top_term_mat)

        # topic_terms
        self.write_topic_terms(top_term_mat)

        # doc-term (STD)
        self.write_doc_term(top_term_mat)

        # topic_topic
        self.write_topic_topic(top_term_mat)

        # term_term
        self.write_term_term(top_term_mat)

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

        # create indices for fast lookup
        self.create_db_indices()
            
    
