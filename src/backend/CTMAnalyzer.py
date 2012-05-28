from itertools import combinations
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
         cmd = '%(ctmdir)s/ctm %(type)s %(corpusfile)s %(ntopics)i %(init)s %(outdir)s %(outdir)s/%(settingsfile)s' % self.params       
         print cmd
         stime = time()
         os.system(cmd)
         print 'finished CTM analysis in %f seconds' % (time()-stime)     
         
#    def create_browser_db(self):
#        """
#
#        """
#        # put the CTM output data into the correct format for the browser
#        if self.params['nterms'] == -1 or self.params['ndocs'] == -1:
#            print 'Must set parameter "nterms" and "ndocs" before calling createBrowser with CTM' # TODO lighten this requirement?
#        bfile = open('%(outdir)s/final-log-beta.dat' % self.params, 'r')
#        boutfile = open('%(outdir)s/final.beta' % self.params, 'w')
#        for tnum in xrange(self.params['ntopics']):
#            for wnum in xrange(self.params['nterms']): # TODO add try catch in case param is set wrong
#                boutfile.write('%f ' % float(bfile.readline()))
#            boutfile.write('\n')
#        boutfile.close()
#        bfile.close()
#
#        gfile = open('%(outdir)s/final-lambda.dat' % self.params, 'r')    # lambda file has documents as the rows and topics as the columns
#        goutfile = open('%(outdir)s/final.gamma' % self.params, 'w')
#        for dnum in xrange(self.params['ndocs']):
#            for wnum in xrange(self.params['ntopics']):
#                val = float(gfile.readline())
#                if not val:
#                    val = -100
#                goutfile.write('%f ' % math.exp(val))
#            goutfile.write('\n')
#            pass
#
#        goutfile.close()
#        gfile.close()
#
#        return super(CTMAnalyzer, self).create_browser_db()

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

        # doc_doc
        lam = np.loadtxt(os.path.join(self.params['outdir'],'final-lambda.dat'))
        lam.shape = (len(lam)/self.params['ntopics'], self.params['ntopics'])
        lam = lam[:,:-1]
        nu = np.loadtxt(os.path.join(self.params['outdir'],'final-nu.dat'))
        nu.shape = (len(nu)/self.params['ntopics'], self.params['ntopics'])
        nu = nu[:,:-1]
        theta_est = []

        for i in xrange(len(lam)):
            samples = logistic_normal(lam[i,:], np.diag(nu[i,:]), n=100) # n=100 found adequate through aux experiments
            theta_est.append(np.sqrt(samples).mean(axis=0))

        scores = np.zeros([len(lam), len(lam)])
        for combo in  combinations(xrange(len(lam)), 2): # generator for all possible doc combinations
            scores[combo[0],combo[1]] = 1/(2 - 2 * sum(theta_est[combo[0]] * theta_est[combo[1]] )) # make score inverse Hellinger so higher is better
        scores = scores + scores.T # for accurate top K doc-docs TODO is there a better way to do this?
        score_inds = self._get_rev_srt_ind((scores))[:,:50] # take the top fifty related docs

        db_list = []
        idxs = {} # so we don't have duplicates in the database
        for i in xrange(scores.shape[0]):
            for j in score_inds[i,:]:
                j = int(j)
                minv = min(i,j)
                maxv = max(i,j)
                if not idxs.has_key('%i %i' % (minv,maxv)):
                        db_list.append((minv, maxv, round(scores[minv,maxv], 4))) # TODO this could probably be replaced with a generator
                        idxs['%i %i' % (minv,maxv)] = 1


        self.ins_docdoc(db_list)

        # doc_topic
        self.write_doc_topic(np.array(theta_est))

        # create indices for fast lookup
        self.create_db_indices()



           

    