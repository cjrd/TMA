from math import exp
from TMAnalyzer import TMAnalyzer
import os  
from time import time
import pdb

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
            'alpha':'estimate', 'outdir':'ldaout', 'settingsfile':'settings.txt', 'infname':'', 'wordct':-1, 'alpha_tech':'estimate'}

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
            cmd = '%(ldadir)s/lda est %(alphaval)f %(ntopics)d %(outdir)s/%(settingsfile)s %(corpusfile)s %(init)s %(outdir)s' % self.params
        else:
            infname = 'inf'
            if self.params.has_key("infname"):
                infname = self.params["infname"]
            self.params['infname'] = os.path.splitext(self.params['outdir'])[0] + '/' + infname
            cmd = '%(ldadir)s/lda inf %(outdir)s/%(settingsfile)s %(outdir)s/final %(corpusfile)s %(infname)s' % self.params

        print '\n' + cmd + '\n'
        stime = time()
        os.system(cmd)
        print 'finished LDA analysis in %f seconds' % (time()-stime)

    def create_browser_db(self):
        return super(LDAAnalyzer, self).create_browser_db()
        
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
    
    def kf_perplexity(self, trainf_list, testf_list, test_wc, param='ntopics', start = -1, stop = -1, step = 5):
        """
        Calculates the LDA perplexity given the training and testing files in trainf_list and testf_list, respectively
        @param trainf_list: list of paths of the training corpora
        @param testf_list: list of paths of the corresponding testing corpora (must be same length as trainf_list)
        @param test_wc: the total number of words in each test corpus
        @return the perplexity for each fold
        """
        # TODO much of this could be parallelized with the appropriate resources
        # TODO fix repititious use of settings files (perhaps fix the c code so we can just pass these params)
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
        train_params = {'type':'est'}
        test_params = {'type':'inf'}
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

                test_params["infname"] = "inf" + str(i)
                test_params["corpusfile"] = testf_list[i]
                self.set_params(test_params)
                self.do_analysis()
                lhoods = open(self.params['infname'] + '-lda-lhood.dat').readlines() # TODO add error catching in case there was an inference problem
                ppts[i].append([param_val, round(exp((sum(map(lambda x: float(x.strip()), lhoods))*-1.0)/test_wc[i]), 3)])
        self.set_params({'outdir':orig_outdir, 'corpusfile':orig_corpusfile}) # return the analyzer to its original state
        return ppts


   
        
            
            
            
    
        