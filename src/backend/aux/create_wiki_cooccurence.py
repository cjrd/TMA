#!/usr/bin/env python
import pdb 
import os 
import re 
import cPickle as pickle
from src.backend.tma_utils import TextCleaner, ids_to_key 
from lib.porter2 import stem
import sqlite3 as sqlite                
from time import time  
import bsddb  
import random

def db_transfer(termterm_dict, termterm_db):
    for t1 in termterm_dict:
        for t2 in termterm_dict[t1]:
            ikey = '%i,%i' % (t1,t2)
            if termterm_db.has_key(ikey):
                termterm_db[ikey] = str(int(termterm_db[ikey]) + termterm_dict[t1][t2])
            else:
                termterm_db[ikey] = str(termterm_dict[t1][t2])


if __name__ == '__main__':
    # DATA
    outdata_dir = '/Users/cradreed/Research/TMBrowse/develarea/'
    wikfile = outdata_dir + 'enwiki_abstracts-20120307.dat' #'/Users/cradreed/Research/TMBrowse/develarea/enwiki-latest-abstract18.xml'#

    # use bsd to create to cooccurence file then write to sqlite to maintain database consistency and reduce dependencies
    # set up the dbase
    #dbfile = '/Users/cradreed/Research/TMBrowse/develarea/wiki-terms.sqlite'
    # wikivocab_file = outdata_dir + 'wikivocab_full.bdb'
    # wikivocab_ct_file =  outdata_dir + 'wikivocab_ct_full.bdb'
    wiki_termterm_file = outdata_dir + 'wiki_termterm_full_100percent.bdb'

    # os.remove(dbfile) # TESTING
    # if os.path.exists(wikivocab_file):
    #     os.remove(wikivocab_file)
    # if os.path.exists(wikivocab_ct_file):
    #     os.remove(wikivocab_ct_file)
#    if os.path.exists(wiki_termterm_file):
#        os.remove(wiki_termterm_file)

    vocab_dict = pickle.load(open(outdata_dir + 'wiki_vocab_dic_full_100percent.obj','rb'))#{}#bsddb.hashopen(wikivocab_file)
    vocab_ct_dict = pickle.load(open(outdata_dir + 'wiki_vocab_ct_full_100percent.obj','rb'))#{}#bsddb.hashopen(wikivocab_ct_file)
    termterm_db = bsddb.btopen(wiki_termterm_file)
    termterm_dict = {}
    text_cleaner = TextCleaner(stopword_file='/Users/cradreed/Research/TMBrowse/trunk/src/backend/aux/stop_words.txt')

    # add the cooccurence information to the table
    st_time = time()
    num_ab = 0
    term_ct = 0
    tot_ab_len = 0
    dep_no = 0
    print_no = 50000
    transfer_no = 10*print_no
    ltime = time()
    with open(wikfile,'r') as wikxml:
        for i, line in enumerate(wikxml):
            # only sample % 20
#            if random.random() > 0.20:
#                continue
            num_ab += 1
            if num_ab <= 3500000:
                continue

            if num_ab % print_no == 0 and not num_ab == 0:
                print 'Parsed {0:8d} of 3925809 abstracts; last {1:5d} abstracts took {2:0.1f} seconds. Average {3:4d} terms per doc.'.format(num_ab, print_no,time()-ltime, int(tot_ab_len/print_no))
                ltime = time()
                tot_ab_len = 0

                if num_ab % transfer_no == 0 and not num_ab == 0:
                    print '---- Transfering %i abstracts to db -----' % transfer_no
                    db_transfer(termterm_dict, termterm_db)
                    dep_no += 1
                    del(termterm_dict)
                    termterm_dict = {}
                    print '---- %i transfer complete, took %0.1f seconds ----' % (dep_no, (time() - ltime))
                    ltime = time()
                    
            text = line.strip() # remove the abstract tags
            text = text_cleaner.parse_text(text)
            text = list(set(text))
            tot_ab_len += len(text)

            for nt1, term1 in enumerate(text):
                if not vocab_dict.has_key(term1):
                    t1_id = term_ct
                    vocab_dict[term1] = t1_id
                    vocab_ct_dict[t1_id] = 1
                    term_ct += 1
                else:
                    t1_id = vocab_dict[term1]
                    vocab_ct_dict[t1_id] += 1

                for nt2 in xrange(nt1+1, len(text)): # 173.271281 vs 185s TODO make sure the counting is correct
                    term2 = text[nt2]
                    if not vocab_dict.has_key(term2):
                        t2_id = term_ct
                        vocab_dict[term2] = t2_id
                        vocab_ct_dict[t2_id] = 0 # avoid overcounting
                        term_ct += 1
                    else:
                        t2_id = vocab_dict[term2]

                    t_keys = ids_to_key(t1_id, t2_id)
                    if not termterm_dict.has_key(t_keys[0]):
                        termterm_dict[t_keys[0]] = {t_keys[1]:1}
                    elif termterm_dict[t_keys[0]].has_key(t_keys[1]):
                        termterm_dict[t_keys[0]][t_keys[1]] += 1
                    else:
                        termterm_dict[t_keys[0]][t_keys[1]] = 1
        db_transfer(termterm_dict, termterm_db)

    print 'Added %i terms to dic' % len(vocab_dict)
    # vocab_dict.close()
    # vocab_ct_dict.close()
    # print termterm_db
    # print vocab_dict
    # print vocab_ct_dict
    termterm_db.close()
    pickle.dump(vocab_dict, open(outdata_dir + 'wiki_vocab_dic_full_100percent2.obj','wb'))
    pickle.dump(vocab_ct_dict, open(outdata_dir + 'wiki_vocab_ct_full_100percent2.obj','wb'))


    time_parse = time() - st_time
    print 'Parsing %i abstracts took %f seconds' % (num_ab, time_parse)