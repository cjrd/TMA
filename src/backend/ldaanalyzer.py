from math import exp
from tmanalyzer import TMAnalyzer
import os  
from time import time
import pdb
import numpy as np

class LDAAnalyzer(TMAnalyzer):
    def __init__(self, params):
        """                                          
         init [rand OR seeded OR a previous beta file] : topic initialization technique (default='rand')
         data [filename]: location of corpus.dat file (default './corpus.dat')                                                                            
         alphaval[float] : initial /fixed alpha value (default = 1.0)
         type: ['est' or 'inf']: ESTimate the model or do INFerence using a previous model (default='est')
         ntopics [integer]: number of topics in LDA model (default=10)
         var_max_iter [integer e.g. 10 or -1 for all]: maximum number of iterations in the e-step fixed-point method for each document (default 20)
         var_convergence [float e.g., 1e-8]: convergence metric limit for fixed point method in the e-step for each document (default 1e-6)
         em_max_iter [integer e.g., 100]: max number of EM iterations (default 100)
         em_convergence [float e.g., 1e-5]: convergence metric limit for the EM algo    rithm (default 1e-4)
         alpha ['fit' OR 'estimate']: fix alpha or estimate it (default 'estimate')
         outdir [filename]: output directory for LDA analysis (default = './ldaout')
        """   
                                                               
        # replace default parameters with pased-in parameters
        ldaparams = {'type':'est', 'alphaval':1.0, 'ntopics':10, 'titlesfile':'titles.txt', 
            'corpusfile':'corpus.dat', 'vocabfile':'vocab.txt', 'init':'random',
            'ldadir':'./','var_max_iter':20, 'var_convergence':1e-6, 'em_max_iter':100, 'em_convergence':1e-4,
            'alpha':'estimate', 'outdir':'ldaout', 'settingsfile':'settings.txt', 'infname':'', 'wordct':-1, 'alpha_tech':'estimate', 'timelimit':-1}

        for prm in params.keys():
            if ldaparams.has_key(prm):
                ldaparams[prm] = params[prm]
            else:
                raise Exception("unkown parameter value for LDAAnalyzer: %s" % prm)
        ldaparams['alg'] = 'lda'       
        super(LDAAnalyzer, self).__init__(ldaparams)
    
    def do_analysis(self):

        setpath = '%(outdir)s/%(settingsfile)s' % self.params
        if not os.path.exists(setpath):
            setfile =  open(setpath, 'w')
            settingsout = "var max iter %(var_max_iter)d\n\
            var convergence %(var_convergence)f\n\
            em max iter %(em_max_iter)i\n\
            em convergence %(em_convergence)f\n\
            alpha %(alpha)s" % self.params
            setfile.write(settingsout)
            setfile.close()

        if self.params['type'] == 'est':
            cmd = 'ulimit -t %(timelimit)d; %(ldadir)s/lda est %(alphaval)f %(ntopics)d %(outdir)s/%(settingsfile)s %(corpusfile)s %(init)s %(outdir)s' % self.params
        else:
            infname = 'inf'
            if self.params.has_key("infname"):
                infname = self.params["infname"]
            self.params['infname'] = os.path.splitext(self.params['outdir'])[0] + '/' + infname
            cmd = 'ulimit -t %(timelimit)d; %(ldadir)s/lda inf %(outdir)s/%(settingsfile)s %(outdir)s/final %(corpusfile)s %(infname)s' % self.params

        print '\n' + cmd + '\n'
        stime = time()
        os.system(cmd)
        print 'finished LDA analysis in %f seconds' % (time()-stime)

    def kf_perplexity(self, trainf_list, testf_list, test_wc, param='ntopics', start = -1, stop = -1, step = 5):
        """
            Calculates the LDA perplexity given the training and testing files in trainf_list and testf_list, respectively
            @param trainf_list: list of paths of the training corpora
            @param testf_list: list of paths of the corresponding testing corpora (must be same length as trainf_list)
            @param test_wc: the total number of words in each test corpus
            @return the perplexity for each fold
        """
        def _handle_ppt(self, test_wci):
            lfile = self.params['infname'] + '-lda-lhood.dat'
            lhoods = open(lfile).readlines()
            return round(((sum(map(lambda x: float(x.strip()), lhoods))*-1.0)/test_wci), 3)

        train_params = {'type':'est'}
        test_params = {'type':'inf'}
        ppts = self.general_kf_ppt(trainf_list, testf_list, test_wc, param, start, stop, step, train_params, test_params, _handle_ppt)
        return ppts

    def createJSLikeData(self):
        # transform the likelihood data
        linfile = open('%s/likelihood.dat' % self.params['outdir'], 'r')
        ldata = linfile.readlines()
        linfile.close()
        jsout = open('%s/js_likelihood.dat'% self.params['outdir'],'w')
        jsout.write('[')
        for i, line in enumerate(ldata):
            lik = line.strip().split()[0]
            jsout.write(lik)
            if not i == len(ldata)-1:
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

        # write topics, i.e. top 3 terms (STD)
        beta = np.loadtxt(os.path.join(self.params['outdir'],'final.beta'))


        # topic_terms - NOTE should be negative valued
        self.write_topic_terms(beta)

        # topic_topic
        self.write_topic_topic(np.exp(beta))

        # term_term
        self.write_term_term(np.exp(beta))

        # doc_doc creation
        gamma = np.loadtxt(os.path.join(self.params['outdir'],'final.gamma'))
        theta = gamma / gamma.sum(1)[:,np.newaxis]
        self.write_doc_doc(theta**0.5)

        # doc_topic
        self.write_doc_topic(theta)

        # write topics
        self.write_topics_table(top_term_mat=beta, doc_top_mat=theta)

        # create indices for fast lookup
        self.create_db_indices()




   
        
            
            
            
    
        