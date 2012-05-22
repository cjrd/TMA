import cPickle as pickle
import pdb  
from bsddb.db import DBNotFoundError
import bsddb 
import sqlite3 as sqlite
# Takes the BDB and converts the info to sqlite (so we don't require multiple DB type) 

def insert_many_coocc(cursor, values_to_insert):  
    cursor.executemany("""
        INSERT INTO  co_occ('term1', 'term2', 'occ')
        VALUES (?, ?, ?)""", values_to_insert)

def insert_many_dic(cursor, values_to_insert):  
    cursor.executemany("""
        INSERT INTO  dict('term', 'id', 'occ')
        VALUES (?, ?, ?)""", values_to_insert)    
        
    

# where's the data from create_wiki_cooccurence.py?
outdata_dir = '/Users/cradreed/Research/TMBrowse/develarea/'
wiki_termterm_file = outdata_dir + 'wiki_termterm_full_100percent2.bdb'
vocab_dict_loc = outdata_dir + 'wiki_vocab_dic_full_100percent2.obj'
vocab_dict_ct_loc = outdata_dir + 'wiki_vocab_ct_full_100percent2.obj'
                
# prep the tables
dbfile = outdata_dir + 'wiki_cocc_100percent.sqlite'
con = sqlite.connect(dbfile)
cur = con.cursor()
# need a dictionary table and a co_occ table                                            
cur.execute('DROP TABLE IF EXISTS dict')
cur.execute('DROP TABLE IF EXISTS co_occ')

cur.execute('CREATE TABLE dict(term VARCHAR PRIMARY KEY, id INTEGER, occ INTEGER)')
cur.execute('CREATE TABLE co_occ(id INTEGER PRIMARY KEY, term1 INTEGER, term2 INTEGER, occ INTEGER)')  
con.commit()

# set up the bdb data and load the dictionaries   
vocab_dict = pickle.load(open(vocab_dict_loc,'rb'))
vocab_ct_dict = pickle.load(open(vocab_dict_ct_loc,'rb'))   
termterm_db = bsddb.btopen(wiki_termterm_file,'r')  

# insert term term
insert_vals = []
ct = 0    
insert_ct = 100000
try: 
    item = termterm_db.first() 
    while True:  
        key = item[0]
        val = item[1]
        ids = key.split(',')
        insert_vals.append((int(ids[0]), int(ids[1]), int(val)))
        ct += 1
        if ct % insert_ct == 0:
            insert_many_coocc(cur, insert_vals)   
            con.commit() 
            insert_vals = [] 
            print 'inserted %i vals' % ct
        item = termterm_db.next()    
except DBNotFoundError:
    insert_many_coocc(cur, insert_vals)   
    con.commit()
    print 'finished processing %i records' % ct
    termterm_db.close() 
        
# insert vocab and count:
insert_vals = [] 
ct = 0
for term in vocab_dict:     
    tid = vocab_dict[term]
    insert_vals.append((term,int(tid), int(vocab_ct_dict[tid])))  
    ct += 1
    if ct % insert_ct == 0:
        insert_many_dic(cur, insert_vals)
        con.commit() 
        insert_vals = [] 
        print 'inserted %i term vals' % ct      
insert_many_dic(cur, insert_vals)
con.commit()

        
# create the indices
print 'creating indices...'
cur.execute('CREATE INDEX dict_idx ON dict(term)')
cur.execute('CREATE INDEX dict_occ ON dict(occ)')
print 'finished dict indices'
cur.execute('CREATE INDEX coocc_idx1 ON co_occ(term1)')
print 'finished coocc_idx1'
cur.execute('CREATE INDEX coocc_idx2 ON co_occ(term2)')
print 'finished coocc_idx2'
cur.execute('CREATE INDEX coocc_idx_occ ON co_occ(occ)')
print 'finished coocc_idx_occ'
cur.execute('CREATE INDEX coocc_idx_all ON co_occ(term1,term2,occ)')
print 'finished coocc_idx_occ'
print 'finished!'

# clean up
con.commit()
con.close()


