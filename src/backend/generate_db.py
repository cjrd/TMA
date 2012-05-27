import sys
import math
import sqlite3
import pdb    
import os   
import cPickle as pickle

### score functions ###
# colo: do the topic scores take the score out of log mode?  

# FOR LDA
# doc_doc: lower is better
# doc_topic: higher is better
# topic_term: higher is better
# topic_topic: lower is better
# term_term: lower is better
# term_topic: higher is better       


# DOCS
def get_doc_score(doca, docb):
    score = 0
    total = 0
    for topic_id in xrange(len(doca)):
        thetaa = doca[topic_id]
        thetab = docb[topic_id]
        if not ((thetaa != 0.0 and thetab == 0.0) or (thetaa == 0.0 and thetab != 0.0)): # TODO this is comparing floats?
            score += math.pow(thetaa - thetab, 2)
    return 0.5 * score


# TOPICS
def get_topic_score(topica, topicb):
    # len(topic) = size of vocabulary
    score = 0
    total = math.pow(abs(math.sqrt(100) - math.sqrt(0)), 2) * len(topica) # TODO what in the world is this???
    for term_id in xrange(len(topica)):
        thetaa = abs(topica[term_id]) # why is this called thata when its taken from the beta file?
        thetab = abs(topicb[term_id]) # furthermore, these are log(beta), so why take abs?
        score += math.pow(abs(math.sqrt(thetaa) - math.sqrt(thetab)), 2) # why take abs before squaring?
    return 0.5 * score / total

# def get_term_score(terma, termb):
#     score = 0
#     for term_id in xrange(len(terma)):
#         score += math.pow(terma[term_id] - termb[term_id], 2)
#     return score  


### write relations to db functions ###
           
# essentially seems to be computing the sse of the different thetas
def write_doc_doc(con, cur, gamma_file):
    cur.execute('CREATE TABLE doc_doc (id INTEGER PRIMARY KEY, doc_a INTEGER, doc_b INTEGER, score FLOAT)')
    con.commit()
    
    # for each line in the gamma file
    read_file = file(gamma_file, 'r')
    docs = []
    for doc in read_file:
        docs.append(map(float, doc.split()))
    read_file.close()
    for i in xrange(len(docs)):
        for j in xrange(len(docs[i])):
            docs[i][j] = math.pow(docs[i][j], 2)
    
    print len(docs)
    for a in xrange(len(docs)):
        if a % 1000 == 0:
            print "doc " + str(a)
        doc_by_doc = {}
        for b in xrange(a, len(docs)): # TODO why compute doc-doc score with itself, should we go from a+1?
            score = get_doc_score(docs[a], docs[b])
            if score == 0: # TODO this is bad: comparing floats..
                continue  # only include the docs if they are not exactly the same (also a sanity check...)?
            elif len(doc_by_doc) < 100: # save the top 100 related docs
                doc_by_doc[score] = (a, b)
            else:
                max_score = max(doc_by_doc.keys())   
                if max_score > score:   # only add the document if it's less than the max score
                    del doc_by_doc[max_score]
                    doc_by_doc[score] = (a, b)
        
        for doc in doc_by_doc:
            execution_string = 'INSERT INTO doc_doc (id, doc_a, doc_b, score) VALUES(NULL, ?, ?, ?)'
            cur.execute(execution_string, [str(doc_by_doc[doc][0]), str(doc_by_doc[doc][1]), str(doc)])
    con.commit()  
    
    cur.execute('CREATE INDEX doc_doc_idx1 ON doc_doc(doc_a)')
    cur.execute('CREATE INDEX doc_doc_idx2 ON doc_doc(doc_b)')
    cur.execute('CREATE INDEX doc_doc_idx_score ON doc_doc(score)')
    con.commit()

 
# directly uses gamma [theta] values
def write_doc_topic(con, cur, gamma_file):
    cur.execute('CREATE TABLE doc_topic (id INTEGER PRIMARY KEY, doc INTEGER, topic INTEGER, score FLOAT)')
    con.commit()
    
    # for each line in the gamma file
    doc_no = 0
    for doc in file(gamma_file, 'r'):
        doc = map(float, doc.split())
        for i in xrange(len(doc)):
            cur.execute('INSERT INTO doc_topic (id, doc, topic, score) VALUES(NULL, ?, ?, ?)', [doc_no, i, doc[i]])
        doc_no = doc_no + 1
    con.commit()
    cur.execute('CREATE INDEX doc_topic_idx1 ON doc_topic(doc)')
    cur.execute('CREATE INDEX doc_topic_idx2 ON doc_topic(topic)')  
    cur.execute('CREATE INDEX doc_topic_idx_score ON doc_topic(score)')
    con.commit()
        

def write_topics(con, cur, beta_file, vocab):
    cur.execute('CREATE TABLE topics (id INTEGER PRIMARY KEY, title VARCHAR(100))')
    con.commit()

    # for each line in the beta file
    indices = range(len(vocab))
    ct = 0
    #topics_file = open(filename, 'a')  # What's this doing?
    for topic in file(beta_file, 'r'):
        topic = map(float, topic.split())
        #indices = range(len(topic))
        indices.sort(lambda x,y: -cmp(topic[x], topic[y]))
        cur.execute('INSERT INTO topics (id, title) VALUES(?, ?)', [ct, buffer("{" + vocab[indices[0]] + ', ' + vocab[indices[1]] + ', ' + vocab[indices[2]] + '}')])
        ct += 1
    con.commit()

    
def write_topic_term(con, cur, beta_file):
    cur.execute('CREATE TABLE topic_term (id INTEGER PRIMARY KEY, topic INTEGER, term INTEGER, score FLOAT)')
    con.commit()
     
    topic_no = 0
    # topic_term_file = open(filename, 'a')
    
    for topic in file(beta_file, 'r'):
        topic = map(float, topic.split())
        indices = range(len(topic)) # note: len(topic) should be the same as len(vocab)
        indices.sort(lambda x,y: -cmp(topic[x], topic[y]))
        for i in xrange(len(topic)):
            cur.execute('INSERT INTO topic_term (id, topic, term, score) VALUES(NULL, ?, ?, ?)', [topic_no, indices[i], topic[indices[i]]])
        topic_no = topic_no + 1
    con.commit()               
    
    cur.execute('CREATE INDEX topic_term_idx1 ON topic_term(topic)')
    cur.execute('CREATE INDEX topic_term_idx2 ON topic_term(term)') 
    cur.execute('CREATE INDEX topic_term_score ON topic_term(score)')
    con.commit()


def write_topic_topic(con, cur, beta_file):
    cur.execute('CREATE TABLE topic_topic (id INTEGER PRIMARY KEY, topic_a INTEGER, topic_b INTEGER, score FLOAT)')
    con.commit()  
    
    # for each line (topic with length V) in the beta file
    read_file = file(beta_file, 'r')
    topics = []
    topic_no = 0
    for topic in read_file:
        topics.append(map(float, topic.split()))
        topic_no += 1
    
    topica_count = 0
    topic_by_topic = []
    for topica in topics:
        #topic_sim = r
        topicb_count = 0
        for topicb in topics:
            if topic_by_topic.count((topicb_count, topica_count)): # make sure we don't compare topics > 1 time
                topicb_count += 1
                continue
            score = get_topic_score(topica, topicb)
            cur.execute('INSERT INTO topic_topic (id, topic_a, topic_b, score) VALUES(NULL, ?, ?, ?)', [topica_count, topicb_count, score])
            
            topic_by_topic.append((topica_count, topicb_count))
            topicb_count += 1
        topica_count += 1
    con.commit() 
    
    cur.execute('CREATE INDEX topic_topic_idx1 ON topic_topic(topic_a)')
    cur.execute('CREATE INDEX topic_topic_idx2 ON topic_topic(topic_b)')
    cur.execute('CREATE INDEX topic_topic_score ON topic_topic(score)')
    con.commit()



def create_term_term_object(out_obj_file, beta_file, no_vocab):
    v = [[] for x in xrange(no_vocab)]
    # pdb.set_trace()
    # cur.execute('CREATE TABLE term_term (id INTEGER PRIMARY KEY, term_a INTEGER, term_b INTEGER, score FLOAT)')
    # con.commit()
    #                  
    # for i in xrange(no_vocab):
    #       v[i] = [] # for number of vocab {i:[]} (a dict of arrays)
    #                                                                 
    # extracts the sqrt(exp(beta)) for each term   
    
    for topic in file(beta_file, 'r'):
        topic = map(float, topic.split()) 
        for i in xrange(no_vocab):
            v[i].append(math.sqrt(math.exp(topic[i]))) # save this as a python object 
    pickle.dump(v,open(out_obj_file, 'wb'))
    return

def write_doc_term(con, cur, wordcount_file, no_words):
    cur.execute('CREATE TABLE doc_term (id INTEGER PRIMARY KEY, doc INTEGER, term INTEGER, score FLOAT)')
    con.commit()
    
    doc_no = 0
    for doc in file(wordcount_file, 'r'):
        doc = doc.split()[1:]
        terms = {}
        for term in doc:
            terms[int(term.split(':')[0])] = int(term.split(':')[1]) # create a term:ct dictionary for each document

        for i in xrange(no_words):
            if terms.has_key(i):
                score = terms[i]
                execution_string = 'INSERT INTO doc_term (id, doc, term, score) VALUES(NULL, ?, ?, ?)'
                cur.execute(execution_string, [doc_no, i, score])

        doc_no += 1      
    con.commit() 
    
    cur.execute('CREATE INDEX doc_term_idx1 ON doc_term(doc)')
    cur.execute('CREATE INDEX doc_term_idx2 ON doc_term(term)')
    cur.execute('CREATE INDEX doc_term_score ON doc_term(term)') 
    con.commit()

def write_terms(con, cur, terms_file):
    cur.execute('CREATE TABLE terms (id INTEGER PRIMARY KEY, title VARCHAR(100))')
    con.commit()

    ct = 0
    for line in open(terms_file, 'r'):
        cur.execute('INSERT INTO terms (id, title) VALUES(?, ?)', [ct,buffer(line.strip())])
        ct += 1

    con.commit()

def write_docs(con, cur, docs_file):
    cur.execute('CREATE TABLE docs (id INTEGER PRIMARY KEY, title VARCHAR(100))')
    con.commit()

    ct = 0
    for line in open(docs_file, 'r'):
        cur.execute('INSERT INTO docs (id, title) VALUES(?, ?)', [ct, buffer(line.strip())])
        ct += 1

    con.commit()


def generate_db(filename, doc_wordcount_file, beta_file, gamma_file, vocab_file, doc_file):
    
    # connect to database, which is presumed to not already exist
    con = sqlite3.connect(filename)
    cur = con.cursor()

    # pre-process vocab, since several of the below functions need it in this format
    vocab = file(vocab_file, 'r').readlines()
    vocab = map(lambda x: x.strip(), vocab)

    # write the relevant rlations to the database, see individual functions for details \
    
    # write the actual terms to the db
    print "writing terms to db..."
    write_terms(con, cur, vocab_file)
    
    # write the docment titles to the db: the doc_id is in order of the titles
    print "writing docs to db..."
    write_docs(con, cur, doc_file)
    
    # compute the sse of gamma  for the documents 
    print "writing doc_doc to db..."
    write_doc_doc(con, cur, gamma_file)
    
    # directly write the gamma values for each document
    print "writing doc_topic to db..."
    write_doc_topic(con, cur, gamma_file)
    
    # write the top 3 terms for each topic of each document
    print "writing topics to db..."
    write_topics(con, cur, beta_file, vocab)
    
    # sort the terms by their beta values for each topic
    print "writing topic_term to db..."
    write_topic_term(con, cur, beta_file)
          
    # sse for the topics
    print "writing topic_topic to db..."
    write_topic_topic(con, cur, beta_file)
    
    # sse across topics for the terms O(V^2) calculation -- this is THE bottleneck
    print "creating term_term obj..."
    #write_term_term(con, cur, beta_file, len(vocab)) 
    term_obj = os.path.join(os.path.dirname(filename), 'term_betas.obj')
    create_term_term_object(term_obj, beta_file, len(vocab))
       
    # doc-term appears to be how many times the term was in the document (should it be relative? -- onlh if we're comparing)
    print "writing doc_term to db..."
    write_doc_term(con, cur, doc_wordcount_file, len(vocab))

### main ### 
if __name__ == '__main__': 
    if (len(sys.argv) != 7):
       print 'usage: python generate_db.py <db-filename> <doc-wordcount-file> <beta-file> <gamma-file> <vocab-file> <doc-file>\n'
       sys.exit(1)

    filename = sys.argv[1]
    doc_wordcount_file = sys.argv[2]
    beta_file = sys.argv[3]
    gamma_file = sys.argv[4]
    vocab_file = sys.argv[5]
    doc_file = sys.argv[6]
    
    generate_db(filename, doc_wordcount_file, beta_file, gamma_file, vocab_file, doc_file)
