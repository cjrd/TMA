from TMAnalyzer import TMAnalyzer
from src.settings import SRC_PATH
from time import time
import os
import math

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
        dbase = self.init_rel_db()

        # write the vocab to the database (STD)

        # write doc title to database (STD)

        # write topics, i.e. top 3 terms (STD)

        # doc-term (STD)

        # doc_doc (doc1, doc2, score

        # calculate the hellinger distance using an R-script
        heldist_cmd = "Rscript %s %s %i" % (os.path.join(SRC_PATH,'backend/aux/hellinger-ctm.r'),self.params['outdir'] + '/' , self.params['ntopics'])
        os.system(heldist_cmd)
        # write the results to a database
        doc_doc_scores = []
        with open(os.path.join(self.params['outdir'], 'hellinger-docs.csv'), 'r') as hdistf:
            for i, doc in enumerate(hdistf):
                dscores = doc.strip().split()
                for j in xrange(len(dscores)):
                    if j > i:
                        doc_doc_scores.append((i, j, 1.0/float(dscores[j])))
        self.ins_docdoc(doc_doc_scores, dbase)
        
        # doc_topic

        # topic_terms

        # topic_topic

        # term_term
        self.create_rel_indices(dbase) # TODO implement



           

    