from TMAnalyzer import TMAnalyzer
import os        
import math     
from time import time

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
            'wordct':-1
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
         cmd = '%(hdpdir)s/hdp --data %(corpusfile)s --algorithm %(algorithm)s --directory %(outdir)s\
 --max_iter %(max_iter)d --save_lag %(save_lag)d --init_topics %(init_topics)i --gamma_a %(gamma_a)f\
 --gamma_b %(gamma_b)f --alpha_a %(alpha_a)f --alpha_b %(alpha_b)f --sample_hyper %(sample_hyper)s\
 --eta %(eta)f --split_merge %(split_merge)s --restrict_scan %(restrict_scan)s' % self.params
         print cmd
         stime = time()
         os.system(cmd)
         print 'finished HDP analysis in %f seconds' % (time()-stime)     
         
    def create_browser_db(self):
        # process and write "beta file" 
        if self.params['ndocs'] == -1:
            print 'Must set parameter "ndocs" before calling create_browser_db with HDP' # TODO lighten this requirement?
        mtfile = open('%(outdir)s/mode-topics.dat' % self.params, 'r')
        betafile = open('%(outdir)s/final.beta' % self.params, 'w') 
        ntopics = 0
        for topic in mtfile:
            ntopics += 1
            terms = map(float, topic.strip().split())
            totprob = sum(terms)
            for trm in terms:
                trm += 1 # TODO: better smoothing technique?
                betafile.write('%f ' % (math.log(trm/totprob)))
            betafile.write('\n')
        mtfile.close()
        betafile.close()
        
        # pull in mode-word-assignments.dat and convert to "gamma file"
        gamma = [[0 for i in range(ntopics)] for j in range(self.params['ndocs'])]
        mwafile = open('%(outdir)s/mode-word-assignments.dat' % self.params, 'r')
        mwafile.readline() # removes header
        for doc in mwafile:
            stats = map(int, doc.strip().split())   
            gamma[stats[0]][stats[2]] += 1 # document is in 0th spot, topic is in 3rd spot
        mwafile.close()    

        # write "gamma file"
        gammafile = open('%(outdir)s/final.gamma' % self.params, 'w')
        for doc in gamma:  
            for doctop in doc:
                gammafile.write('%d ' % doctop)
            gammafile.write('\n') 
        gammafile.close() 

        return super(HDPAnalyzer, self).create_browser_db()
        
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
            
    
