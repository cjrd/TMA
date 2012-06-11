from os import path
import cPickle as pickle
import math
import pdb
from src.backend.math_utils import hellinger_distance
from src.backend.tma_utils import slugify
import numpy as np


class Document:
    """
    Class to represent the documents by title and id
    """

    def __init__(self, doc_id, title):
        self.id = doc_id
        self.title = unicode(str(title),errors='ignore')


    def __hash__(self):
        return hash((self.id, self.title))


    def __eq__(self, other):
        return (self.id, self.title) == (other.id, other.title)


    def get_safe_title(self):
        safe_title = slugify(self.title)
        return safe_title


class Topic:
    """
    Class to represent the topics obtained from the analyzer
    """

    max_score = -1
    def __init__(self, rel, topic_id, title, score = -1):
        self.rel = rel
        self.topic_id = topic_id 
        self.id = topic_id 
        self.title = unicode(str(title),errors='ignore')
        self.terms = {}
        self.ranked_terms = []
        self.term_score_total = 0
        self.score = score


    def __hash__(self):
        return hash((self.id, self.title))


    def __eq__(self, other):
        return (self.id, self.title) == (other.id, other.title)


    def get_term(self, rank): # TODO this may be unneeded  and possibly inaccurate
        """
        obtain the rank-th most likely term for the given topic
        """
        if self.terms == {} or rank >= len(self.ranked_terms):
            self.terms = self.rel.get_topic_terms(self)
            self.ranked_terms = sorted(self.terms, key=self.terms.get, reverse=True)
        
        if rank >= len(self.ranked_terms):
            return None
        
        return self.ranked_terms[rank]


    def get_terms(self, cutoff=-1):
        """
        Obtain the top cutoff terms for the given topic (default=all terms)
        @param cutoff: the number of terms to return
        @return: the top cutoff terms for the given topic (default=all terms)
        """
        if self.terms == {}:
            self.terms = self.rel.get_topic_terms(self, cutoff)
            self.ranked_terms = sorted(self.terms, key=self.terms.get, reverse=True)  
        return self.ranked_terms[:cutoff]


    def get_safe_title(self):
        safe_title = slugify(self.title)
        return safe_title


    def get_terms_list(self, st=0, end = None):
        if self.terms == {}:
            print 'Call topics.get_terms(number) before calling get_terms_string'
        if end is None:
            end = len(self.ranked_terms)
        return map(lambda x: [x.title, x.id], self.ranked_terms[st:end])


class Term:
    """
    Class to represent the  terms by id and title and keep track of obtained terms to avoid excessive DB queries
    """
    all_terms = {} # keep track of the acquired terms to limit db queries
    max_occ = -1

    def __init__(self, term_id, title, count):
        self.id = term_id
        self.title = str(title)
        self.count = count
        Term.all_terms[term_id] = self


    def __hash__(self):
        return hash((self.id, self.title))


    def __eq__(self, other):
        return (self.id, self.title) == (other.id, other.title)


    def get_safe_title(self):
        return slugify(unicode(self.title))

    
    def set_title(self, title):
        self.title = str(title)
        

class relations:
    """
    General class to calculate and determine relations between topics, terms, and documents
    """

    def __init__(self, mydb):
        self.mydb = mydb 
        self.term_topic_obj_loc = path.join(path.dirname(mydb.get_db_loc()), 'top_term_mat.obj')
        self.topics = [] # TODO do we actually need these or are they redundant?
        self.docs = []
        self.terms = []


    def get_term(self, term_id):
        """
        obtain the term corresponding to term_id
        """
        if Term.all_terms.has_key(term_id):
            return Term.all_terms[term_id]
        else: 
            term_qry = self.mydb.get_term(term_id)
            if  term_qry == []:
                return None
            return Term(term_id, term_qry[0][1], term_qry[0][2])


    def get_terms(self, cutoff = -1, start_val = -1, end_val = -1):
        """
        Obtain a list of terms in the specified range or cut off [start_val:end_val] or [:cutoff], resp.
        """
        use_range = (start_val < end_val) and start_val > -1
        if use_range:
            cutoff = end_val

        terms_info = self.mydb.get_term_info(cutoff)
        Term.max_occ = terms_info[0][2]
        if use_range:
            terms_info = terms_info[-(end_val - start_val):]

        for term_info in terms_info:
            term_id = term_info[0]
            term_title = term_info[1]

            if len(term_info) > 2:
                term_count = term_info[2]
            else:
                term_count = -1

            term = Term(term_id, term_title, term_count)
            # add to the global terms list as well
            if not Term.all_terms.has_key(term_id):
                Term.all_terms[term_id] = term
            self.terms.append(term)

        return self.terms


    def get_topics(self, cutoff = -1, start_val = -1, end_val = -1):
        """
        obtain a list of topics from the database
        @return a list of topics from the database default sorted by overall_score, i.e. total likelihood
        """
        use_range = (start_val < end_val) and start_val > -1
        if use_range or cutoff != -1 or self.topics == []:
            self.topics = []
            if use_range:
                cutoff = end_val

            topics_info = self.mydb.get_topics_info(cutoff)
            Topic.max_score = topics_info[0][2]
            if use_range:
                topics_info = topics_info[start_val:end_val]

            for topic_info in topics_info:
                topic_id = topic_info[0]
                title = topic_info[1]
                score = topic_info[2]
                self.topics.append(Topic(self, topic_id, title, score))

        return self.topics


    def get_topic(self, topic_id):
        """
        return the topic corresonding to topic_id
        """
        topic_info = self.mydb.get_topic_info(topic_id )
        if not topic_info:
            return None
        title = topic_info[0][1]
        return Topic(self, topic_id, title)


    def get_docs(self, cutoff = -1, start_val = -1, end_val = -1):
        """
        Obtain a list of docs in the specified range or cut off [start_val:end_val] or [:cutoff], resp.
        """
        use_range = (start_val < end_val) and start_val > -1
        if use_range:
            cutoff = end_val

        if self.docs == [] or (len(self.docs) < end_val != -1):
            docs_info = self.mydb.get_docs_info(cutoff)
            if use_range:
                docs_info = docs_info[-(end_val - start_val):]
            for doc_info in docs_info:
                doc_id = doc_info[0]
                title = doc_info[1]
                self.docs.append(Document(doc_id, title))
        if use_range:
            return self.docs[-start_val:end_val]
        else:
            return self.docs


    def get_doc(self, doc_id):
        """
        Obtain the document corresponding to doc_id
        """
        doc_info = self.mydb.get_doc_info(doc_id )
        title = doc_info[0][1]
        return Document(self, doc_id, title)
    
        
    def get_topic_terms(self, topic, cutoff=-1):
        """
        Obtain the most likely terms for a given topic [:cutoff] where default is to return all terms
        """
        topic_terms_info = self.mydb.get_topic_terms(topic.topic_id, cutoff)
        topic_terms = {}
        for info in topic_terms_info:
            term_id = info[2]
            score = info[3]
            term =self.get_term(term_id)
            if term is not None:
                topic_terms[term] = score
        return topic_terms


    def get_top_related_docs(self, token, num=1):
        """
        Obtain the most likely documents for a given term (token) where default is to return the top document
        """
        token_doc_info = []
        if isinstance(token, Topic):
            token_doc_info = self.mydb.get_top_topic_docs(token.id,num)
        elif isinstance(token, Document):
            token_doc_info = self.mydb.get_top_doc_docs(token.id,num)
        elif isinstance(token, Term):
            token_doc_info = self.mydb.get_top_term_docs(token.id,num)

        token_docs = {}
        for info in token_doc_info:
            doc_id = info[1]
            if isinstance(token, Document) and info[1] == int(token.id):
                doc_id = info[2]
            score = info[3]
            if score: #TODO: is there better way to do this?
                doc_info = self.mydb.get_doc_info(doc_id)
                if doc_info != []:
                    title = doc_info[0][1]
                    token_docs[Document(doc_id, title)] = score
                #if len(token_docs.keys()) > 30:
                    #break

        return token_docs 


    def get_top_related_topics(self, token, num=1):
        """
        get the top related topics for (1) other topics, (2) documents, (3) terms
        """
        token_topic_info = []
        if isinstance(token, Topic):
            token_topic_info = self.mydb.get_top_topic_topics(token.topic_id, num)
        elif isinstance(token, Document):
            token_topic_info = self.mydb.get_top_doc_topics(token.id, num)
        elif isinstance(token, Term):
            token_topic_info = self.mydb.get_top_term_topics(token.id, num)

        topics = {}
        for info in token_topic_info:
            score = info[3]
            if score: #check for reverse pairs in topic-topic search
                if (isinstance(token, Topic) and info[2] == int(token.topic_id)) or isinstance(token, Term):
                    t = self.get_topic(info[1])
                    if t is not None:
                        topics[t] = score #TODO: topic init needs work
                else:
                    t = self.get_topic(info[2])
                    if t is not None:
                        topics[t] = score 
        return topics


    def get_top_related_terms(self, term, top_n = 10):
        """
        Get the top_n terms related to the given term
        """
        term_id = term.id
        top_term_mat = pickle.load(open(self.term_topic_obj_loc,'rb'))
        max_score = 100000000

        # compute the inverse Hellinger distance using the topic distributions for each term  (lower is better)
        term = top_term_mat[term_id,:]
        scores = hellinger_distance(term, top_term_mat)
        scores[term_id] = max_score
        scores = 1/scores
        top_term_ids = np.argsort(scores)[::-1][:top_n]
        top_terms = []
        for ttid in top_term_ids:
            ttid = int(ttid)
            trm = self.get_term(ttid)
            top_terms.append(trm)
        return top_terms

    def get_top_in_term_rel_pct(self, topic, term, *args, **kwargs):
        """
        Obtain the relative percent of the topic in the given term
        """
        topics = self.get_top_related_topics(term, -1)
        return 100*math.exp(topics[topic]) / sum(map(math.exp,topics.values()))


    def get_term_in_top_rel_pct(self, term, topic, *args, **kwargs):
        """
        @return the relative percent of probability assigned to the given term in the topic
        """
        threshold = 0.005 # default value
        if kwargs.has_key('threshold'):
            threshold = kwargs['threshold']

        if not topic.terms.has_key(term):
            topic.get_terms()
        if not topic.term_score_total:
            for t in topic.ranked_terms:
                topic.term_score_total += math.exp(topic.terms[t])

        percent = (math.exp(topic.terms[term]) / topic.term_score_total)

        if percent < threshold:
            return 0
        else:
            return percent*100
