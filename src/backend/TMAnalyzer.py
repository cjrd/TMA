from src.backend.generate_db import generate_db

import os    
from time import time

class TMAnalyzer(object):
    """"Base class for topic model analyzers, e.g. LDAAnalyzer"""
    
    def __init__(self, params): 
        if not params.has_key('outdir'):
            params['outdir'] = '.' # the outdir is required
        if not os.path.exists(params['outdir']):
            os.makedirs(params['outdir'])   
        self.params = params
        
    def do_analysis(self):
        raise NotImplementedError( "Should have implemented do_analysis in subclass analyzer" )
        
    
    def create_browser_db(self):
        print 'generating tmve database'
        stime = time()
        generate_db(
            filename = self.params['outdir'] + '/tmvedb.sqlite', 
            doc_wordcount_file = self.params['corpusfile'] ,
            beta_file = self.params['outdir'] + '/final.beta',
            gamma_file = self.params['outdir'] + '/final.gamma',
            vocab_file = self.params['vocabfile'],
            doc_file = self.params['titlesfile']
        )
        rtime = time() - stime
        print 'finished generating database in  %0.1f seconds' % rtime 
           
    def get_params(self):
        return self.params

    def get_param(self, param):
        return self.params[param]
        
    def set_params(self, dict):
        for prm in dict.keys():
            if self.params.has_key(prm):
                self.params[prm] = dict[prm]
            else:
                raise Exception("unkown parameter value: %s\nNOTE: first initialize each analyzer" % prm)
    