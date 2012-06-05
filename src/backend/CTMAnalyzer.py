from itertools import combinations
from math import exp
from TMAnalyzer import TMAnalyzer
from src.backend.math_utils import logistic_normal
from time import time
import os
import numpy as np
import pdb

class CTMAnalyzer(TMAnalyzer):
    
    def __init__(self, params):
        """
        TODO Describe parameters
        """      
        
        ctmparams = {'type':'est', 'ntopics':10, 'titlesfile':'titles.txt', 
            'corpusfile':'corpus.dat', 'vocabfile':'vocab.txt', 'init':'seed',
            'ctmdir':'./','var_max_iter':20, 'var_convergence':1e-6, 'em_max_iter':100, 'em_convergence':1e-4,  
            'outdir':'ctmout', 'settingsfile':'settings.txt', 'cg_max_iter':-1, 'cg_convergence':1e-6, 'lag':10,
            'covariance_estimate':'mle', 'nterms':-1, 'ndocs': -1, 'wordct':-1}

        for prm in params.keys():
            if ctmparams.has_key(prm):
                ctmparams[prm] = params[prm]
            else:
                raise Exception("unkown parameter value for CTMAnalyzer: %s" % prm)
        ctmparams['alg'] = 'ctm'    
        super(CTMAnalyzer, self).__init__(ctmparams)   
        
        
    
    def do_analysis(self):
         """
         """
         # write settings file to outputdir 
         settingsout = "em max iter %(em_max_iter)i\n\
         var max iter %(var_max_iter)d\n\
         cg max iter %(cg_max_iter)d\n\
         em convergence %(em_convergence)f\n\
         var convergence %(var_convergence)f\n\
         lag %(lag)i\n\
         covariance estimate %(covariance_estimate)s" % self.params
         setfile =  open('%(outdir)s/%(settingsfile)s' % self.params, 'w')
         setfile.write(settingsout)
         setfile.close()                                                                                             
         
         # do the analysis
         if self.params['type'] == 'est':
            cmd = '%(ctmdir)s/ctm %(type)s %(corpusfile)s %(ntopics)i %(init)s %(outdir)s %(outdir)s/%(settingsfile)s' % self.params
         else:
            infname = 'inf'
            if self.params.has_key("infname"):
                infname = self.params["infname"]
            self.params['infname'] = os.path.splitext(self.params['outdir'])[0] + '/' + infname
            cmd = '%(ctmdir)s/ctm inf %(corpusfile)s %(outdir)s/final %(infname)s %(outdir)s/%(settingsfile)s ' % self.params

         print cmd
         stime = time()
         os.system(cmd)
         print 'finished CTM analysis in %f seconds' % (time()-stime)

    def kf_perplexity(self, trainf_list, testf_list, test_wc, param='ntopics', start = -1, stop = -1, step = 5):
        """
            Calculates the CTM perplexity given the training and testing files in trainf_list and testf_list, respectively
            @param trainf_list: list of paths of the training corpora
            @param testf_list: list of paths of the corresponding testing corpora (must be same length as trainf_list)
            @param test_wc: the total number of words in each test corpus
            @return the perplexity for each fold
        """
        def _handle_ppt(self, test_wci):
            lfile = self.params['infname'] + '-ctm-lhood.dat'
            lhoods = open(lfile).readlines()
#            pdb.set_trace()
            ppt = round(((sum(map(lambda x: float(x.strip()), lhoods))*-1.0)/test_wci), 3)
            return ppt

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
        jsout.write('[');                            
        for i, line in enumerate(ldata):   
            lik = line.strip().split()[1]
            jsout.write(lik)
            if not i == len(ldata)-1:
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

        # write topics, i.e. top 3 terms (STD)
        beta = np.loadtxt(os.path.join(self.params['outdir'],'final-log-beta.dat'))
        beta.shape = (self.params['ntopics'], len(beta)/self.params['ntopics'])
        indices = self._get_rev_srt_ind(beta)
        self.write_topics_table(top_term_mat=beta, indices=indices)

        # topic_terms
        self.write_topic_terms(beta)

        # doc-term (STD)
        self.write_doc_term(beta)

        # topic_topic
        self.write_topic_topic(np.exp(beta))

        # term_term
        self.write_term_term(np.exp(beta))

        # doc_doc -- custom for CTM -- TODO perhaps port some of this code to helper methods/functions
        lam = np.loadtxt(os.path.join(self.params['outdir'],'final-lambda.dat'))
        lam.shape = (len(lam)/self.params['ntopics'], self.params['ntopics'])
        lam = lam[:,:-1]
        nu = np.loadtxt(os.path.join(self.params['outdir'],'final-nu.dat'))
        nu.shape = (len(nu)/self.params['ntopics'], self.params['ntopics'])
        nu = nu[:,:-1]
        sqrt_theta = []

        for i in xrange(len(lam)):
            samples = logistic_normal(lam[i,:], np.diag(nu[i,:]), n=100) # n=100 found adequate through aux experiments
            sqrt_theta.append(np.sqrt(samples).mean(axis=0))
        sqrt_theta = np.array(sqrt_theta)
        self.write_doc_doc(sqrt_theta)


        # doc_topic
        self.write_doc_topic(np.array(sqrt_theta))

        # create indices for fast lookup
        self.create_db_indices()



           

    