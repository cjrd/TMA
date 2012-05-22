import bsddb
import cPickle as pickle
       
outdata_dir = '/Users/cradreed/Research/TMBrowse/develarea/'

print 'starting'
dbfile = '/Users/cradreed/Research/TMBrowse/develarea/wiki_termterm_full.bdb'
termterm_db = bsddb.btopen(dbfile,'r')
ct = 0 
add_ct = 0
all_dic = {}  
popular_dic = {} 
minval = 60

for k,v in termterm_db.iteritems():  
    if ct % 10000 == 0:
        print 'On %i item' % ct      
    val = int(v)               
    if minval < val: 
        all_dic[k] = v   
        add_ct += 1
        break   
    ct += 1 
print 'added % i to all dic' % add_ct   
ct = 0
for w in sorted(termterm_db, key=termterm_db.get, reverse=True): 
    popular_dic[w] = all_dic[w]
    del(all_dic[w])
    if ct > 50000:
        break
       
pickle.dump(popular_dic, open(outdata_dir + 'popterm_dic.obj','wb'))          
termterm_db.close()    
print 'done' 



  