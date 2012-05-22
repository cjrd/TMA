from TMAnalyzer import TMAnalyzer
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
         
    def create_browser_db(self):
        # put the CTM output data into the correct format for the browser
        if self.params['nterms'] == -1 or self.params['ndocs'] == -1:
            print 'Must set parameter "nterms" and "ndocs" before calling createBrowser with CTM' # TODO lighten this requirement?
        bfile = open('%(outdir)s/final-log-beta.dat' % self.params, 'r')
        boutfile = open('%(outdir)s/final.beta' % self.params, 'w')
        for tnum in xrange(self.params['ntopics']):
            for wnum in xrange(self.params['nterms']): # TODO add try catch in case param is set wrong
                boutfile.write('%f ' % float(bfile.readline()))
            boutfile.write('\n')     
        boutfile.close()
        bfile.close()                                                     
        
        gfile = open('%(outdir)s/final-lambda.dat' % self.params, 'r')    # lambda file has documents as the rows and topics as the columns 
        goutfile = open('%(outdir)s/final.gamma' % self.params, 'w')
        for dnum in xrange(self.params['ndocs']):   
            for wnum in xrange(self.params['ntopics']):
                val = float(gfile.readline())     
                if not val:
                    val = -100
                goutfile.write('%f ' % math.exp(val))
            goutfile.write('\n')
            pass 
            
        goutfile.close()
        gfile.close()
               
        return super(CTMAnalyzer, self).create_browser_db()
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
            
    
        
if __name__=="__main__":
    prms = {'outdir':'ctmtestout','corpusfile':'/Users/cradreed/Research/TMBrowse/develarea/tmpiOVSwN_formdata/tmpq62vA7_corpus/corpus.dat',\
        'ctmdir':'/Users/cradreed/Research/TMBrowse/current/lib/ctm-dist', 'nterms':3255, 'ndocs':301}       
    ctmtest = CTMAnalyzer(prms) 
    print 'testing analysis'
    # ctmtest.do_analysis()
    print 'finished analysis'
    print
    
    print 'testing browser' 
    ctmtest.createBrowser('/Users/cradreed/Research/TMBrowse/current/lib/tmve');
    print 'finished browser'        

    