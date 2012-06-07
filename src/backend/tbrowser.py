# a class to control the djangoiation of the tbrowser
import shutil
from django.shortcuts import render_to_response
from django.template import Context, RequestContext
from django.http import HttpResponse
from django.utils.html import escape
from settings import BING_API_KEY, WIKI_COCC_DB, WIKI_NUM_ABST
from urllib2 import HTTPError
import math

from src.backend.relations import relations, Document, Topic, Term
from src.backend.db import db
from src.backend.tma_utils import slugify, remove_non_ascii, median
from lib.simplebing.SBing import SBing
from lib.porter2 import stem

import pdb
from math import exp, log
import random
import os
from time import time
import cPickle as pickle

NUM_TERMS = 8 # TODO move this to settings
TOP_TOPIC_OBJ = 'top_topic_terms.obj' # TODO move this to settings

def get_rel_page(request, alg_db, dataloc, alg=''):
    myrelations = relations(alg_db)
    # build the doc-doc   
    ddata = build_graph_json(myrelations.get_docs(), myrelations, myrelations.get_top_related_docs)
    # and term-term graphs       
    tdata =  {}
    # tdata = build_graph_json(myrelations.get_terms(), myrelations, myrelations.get_top_related_terms)

    # acquire the topics
    topdata = build_topic_json(myrelations)
    # feed the results into the heavily customized template        
    ret_val = render_to_response("model-relationships.html", {'alg':alg, 'doc_data':ddata, 'term_data':tdata, 'top_data':topdata}, context_instance=RequestContext(request))

    return ret_val


def build_topic_json(myrelations):
    topdata="["
    topics= myrelations.get_topics()
    for top in topics:
        topdata += '{"id":%i, "title":"%s"},' % (top.id, top.title)
    topdata = topdata[0:len(topdata)-1] + ']'# remove trailing comma
    return topdata


def build_graph_json(collection, myrelations, related_item_fnct):
    node_str = '"nodes":['
    link_str = '"links":['
    id_to_ind = {}
    for i in xrange(len(collection)):
        id_to_ind[collection[i].id] = i

    for ct, item in enumerate(collection):                  # TODO must make the SQL queries more efficient!    
        name = item.get_safe_title() + '-' + str(item.id)
        group = myrelations.get_top_related_topics(item, 1)
        try:
            group = group.keys()[0]
        except IndexError:
            print 'Warning: %s (%i) did not have related topics, will omit.' % (name, item.id)
            continue

        node_str += '{"name":"%s","group":%s}' % (name, str(group.id))
        if not ct == len(collection)-1:
            node_str += ','

        # form links by finding related items (for now take top item for each link)
        ritems = related_item_fnct(item)
        item_keys = ritems.keys()
        item_keys.sort(lambda x, y: -cmp(ritems[x], ritems[y]))
        source = ct;
        # TODO, make multiple options for view graph (based on similarity score value, number of similar items, etc)  
        target = id_to_ind[item_keys[0].id]
        if not target == None:
            link_str += '{"source":%s,"target":%s,"value":%s}' % (str(source), str(target), str(ritems[item_keys[0]]))
            if not ct == len(collection)-1:
                link_str += ','

    node_str += ']'
    link_str += ']'
    data_str = '{' + node_str + ',' + link_str + '}'

    return data_str

def get_doc_text(docloc, title_wID, numbytes=500):
    doc_title = " ".join(title_wID.split('-')[0:-1])
    try:
        doc_text_file = open(os.path.join(docloc, slugify(unicode(doc_title))),'r')
    except IOError:
        doc_text_file = open(os.path.join(docloc, slugify(unicode(doc_title)) + '.txt'),'r') # TODO fix hack
    txt = doc_text_file.read(numbytes)
    doc_text_file.close()
    doc_text = escape(remove_non_ascii(txt))
    doc_text += "...<br /> <div style=\"text-align:center; margin-top: 10px;\"> <input type=\"button\" name=\"b1\" value=\"View full document\" onclick=\"openlw('" + title_wID + "')\" /> </div>"
    return HttpResponse(doc_text)



def get_summary_page(request, alg_db, numterms = NUM_TERMS, numcolumns = 3, alg=''):
    """
    Returns the Analyzer's summary page
    @param request: the django request object
    @param alg_db: the algorithm database
    @param numterms: the number of terms to display for the topics
    @param numcolumns: the number of columns to use to display the topics
    @param alg: the name of the algorithm e.g. 'LDA' or "HDP'
    """
    myrelations = relations(alg_db)
    topics= myrelations.get_topics()

    ncol = min(len(topics), numcolumns) # number of columns for the data table
    disp_tops = []
    topic_row = []

    ct = 1
    for i in xrange(len(topics)):
        topic_trms = map( lambda x: {'title':x.title, 'id':x.id}, topics[i].get_terms(numterms)) # get the top numterms terms
        topic_row.append({'title':topics[i].title, 'id':topics[i].topic_id, 'terms':topic_trms})

        if ct == ncol or i==len(topics)-1: # group nicely into rows for django
            ct = 1
            disp_tops.append(topic_row)
            topic_row = []
        else:
            ct += 1

    template = 'summary.html'

    return render_to_response(template, {'disp_topics':disp_tops, 'alg':alg}, context_instance=RequestContext(request))

def get_model_page(request, alg_db, corpus_dbloc, dataloc, alg='', num_terms=NUM_TERMS, bing_dict={}, form=None): # TODO stem the terms once for bing and wiki calculations
    """
    Obtain/save the appropriate data for the model-description page
    @param request: the django request object
    @param alg_db: the algorithm database
    @param corpus_dbloc: the location of the corpus specific database (contains cooccurence information)
    @param dataloc: the location of the data for this analysis
    @param alg: the algorithm used for this analysis
    @param num_terms: the number of terms to display for the topics
    """

    alg_loc = "%s/%s" % (dataloc, alg)
    # prep the likelihood results
    lin = open(os.path.join(alg_loc,'js_likelihood.dat'), 'r')
    ldata = lin.readline().strip()
    lin.close()

    myrelations = relations(alg_db)
    topics = myrelations.get_topics()
    log_like = {}
    for tpc in topics:
        terms = myrelations.get_topic_terms(tpc, num_terms)
        log_like[tpc.id] = round( sum(terms.values()), 2)
    tc_scores = get_topic_coherence_scores(topics, corpus_dbloc) # vetted for id alignment in array
    for tid in tc_scores:
        tc_scores[tid] = round(median(tc_scores[tid]),2)

    # see if we already acquired the bing scores, if not, save the topic_terms for AJAX queries
    search_title_scores = {}
    if bing_dict != {}:
        search_title_scores= bing_dict
    else:
        save_topic_terms(topics, alg_loc, num_terms)

    wiki_abs_scores = get_wiki_pmi_coherence(topics)
    for tid in wiki_abs_scores:
        if len(wiki_abs_scores[tid]) > 0:
            wiki_abs_scores[tid] = round(median(wiki_abs_scores[tid]), 2)
        else:
            wiki_abs_scores[tid] = -1 # TODO verify why this happens --- it's usually with strange words

    # build "topic dictionaries" for faster django access
    top_dicts = []

    srt_tc_scores = sorted(tc_scores.values(), reverse=False)
    srt_search_title_scores = sorted(search_title_scores.values(), reverse=False)
    srt_wiki_abs_scores = sorted(wiki_abs_scores.values(), reverse=False)
    srt_log_like = sorted(log_like.values(), reverse=False)
    ntopics = len(topics)
    for i in xrange(ntopics):
        top_dicts.append({})
        topics[i].get_terms(num_terms)
        top_dicts[i]['id'] = topics[i].id
        top_dicts[i]['terms'] = ', '.join([topics[i].get_term(x).title for x in range(num_terms)])
        if tc_scores != {}:
            top_dicts[i]['tc_score'] = tc_scores[topics[i].id]
            top_dicts[i]['tc_score_alpha'] = round(srt_tc_scores.index(tc_scores[topics[i].id])/float(ntopics-1),3) # TODO remove code repetion
        if search_title_scores != {}:
            top_dicts[i]['search_title_score'] = search_title_scores[topics[i].id]
            top_dicts[i]['search_title_score_alpha'] = round(srt_search_title_scores.index(search_title_scores[topics[i].id])/float(ntopics-1),3)
        if wiki_abs_scores != {}:
            top_dicts[i]['wiki_abs_score'] = wiki_abs_scores[topics[i].id]
            top_dicts[i]['wiki_abs_score_alpha'] = round(srt_wiki_abs_scores.index(wiki_abs_scores[topics[i].id])/float(ntopics-1),3)
        if log_like != {}:
            top_dicts[i]['topic_likelihood'] = log_like[topics[i].id]
            top_dicts[i]['topic_likelihood_alpha'] = round(srt_log_like.index(log_like[topics[i].id])/float(ntopics-1),3)
    rgb = {"r":255,"g":171,"b":115}
    ret_val = render_to_response("model-converge.html", {'form': form, 'rgb':rgb, 'like_data':ldata, 'topics':top_dicts, "query_bing":search_title_scores=={}}, context_instance=RequestContext(request))
    return ret_val



def save_topic_terms(topics, loc, numterms=NUM_TERMS):
    """
    Save the top numterms terms from each topic into a dictionary on file named TOP_TOPIC_OBJ in loc
    @param topics: the topics obtained from relations.get_topics()
    @param loc: location to store the topic dictionary
    @param numterms: number of terms to save for each topic
    """
    topic_terms = {}
    for i in xrange(len(topics)):
        topic_terms[topics[i].id] = topics[i].get_terms_list(st=0, end=numterms)
    pickle.dump(topic_terms, open(os.path.join(loc, TOP_TOPIC_OBJ),'wb'))


# Coherence from Mimno, 2011 Topic Coherence        
def get_topic_coherence_scores(topics, corpus_dbloc, numterms=NUM_TERMS): # TODO incorporate this with summary page?
    # TODO make sure this calculation is correct
    dbase = db(corpus_dbloc)
    scores = {}#[[] for i in range(len(topics))]
    for i in xrange(len(topics)):
        scores[topics[i].id] = []
        topics[i].get_terms(numterms) # prep the top numterms terms
        for m in xrange(1,numterms):
            for l in xrange(0,m): # [x]range goes to m-1
                dl_set = set(dbase.get_doc_occ(topics[i].get_term(l).id)) # TODO: could optimize the intersection by sorting the sqlite query
                dm_set = set(dbase.get_doc_occ(topics[i].get_term(m).id))
                dl = len(dl_set)
                dml= len(dl_set.intersection(dm_set))
                scores[topics[i].id].append( log(float((dml + 1))/dl))
    del(dbase)
    return scores

# From Newman, 2010 Automatic Evaluation of Topic Models
def get_wiki_pmi_coherence(topics, numterms=NUM_TERMS):   # TODO make sure the terms are already stemmed
    dbase = db(WIKI_COCC_DB)
    if not dbase.check_table_existence('co_occ'):
        return {}
    scores = {}
    rtime = time()
    tid_dict = {} # keep terms and cooccurence counts in memory for caching
    cocc_dict = {}
    for i in xrange(len(topics)):
        scores[topics[i].id] = []
        print 'Determining wikipedia PMI coherence for topic %i of %i; last topic took %0.1fs' % (i,len(topics), time() - rtime)
        rtime = time()

        # prep the top numterms terms
        titles = []
        topics[i].get_terms(numterms)
        for j in xrange(numterms):
            # TODO make sure stemming is handled consistently
            titles.append(stem(topics[i].get_term(j).title))
            if not tid_dict.has_key(titles[-1]):
                res = dbase.get_wiki_occ(titles[-1])
                if res == []: # don't include terms that are not in the database TODO better way to handle this?
                    del(titles[-1])
                    numterms -= 1
                    continue
                tid_dict[titles[-1]] = [res[0], res[1]] # res[0] is the term_id res[1] is the occurance


        for m in xrange(1,numterms):
            tid1 = tid_dict[titles[m]][0]
            t1_occ = tid_dict[titles[m]][1]
            for l in xrange(0,m): # [x]range goes to m-1
                tid2 = tid_dict[titles[l]][0]   ##topics[i].get_term(l).title
                t2_occ = tid_dict[titles[l]][1]
                min_tid = min(tid1,tid2)
                max_tid = max(tid1,tid2)
                # see if we already found the given cooccurence
                db_cocc_lookup = True
                if cocc_dict.has_key(min_tid):
                    if cocc_dict[min_tid].has_key(max_tid):
                        db_cocc_lookup = False
                else:
                    cocc_dict[min_tid] = {}

                if db_cocc_lookup:
                    cocc_dict[min_tid][max_tid] = dbase.get_wiki_cocc(tid1, tid2, min(t1_occ, t2_occ))
                co_occs = cocc_dict[min_tid][max_tid]

                numer = (co_occs + 1)*WIKI_NUM_ABST # +1 is for smoothing
                denom = t1_occ*t2_occ
                scores[topics[i].id].append( log((float(numer))/denom))
    return scores
    # compute PMI using mean/median of log(n(both)/n(w1)n(w2))



def get_bing_coherence_dict(terms_dict, corpus_dbloc, numtitles=50):
    """
    Coherence from Newman, 2011 Automatic (search index with Bing)
    """
    dbase = db(corpus_dbloc)
    # do we have a de-stemming table?                        
    destem = dbase.check_table_existence("termid_to_prestem")
    bing = SBing(BING_API_KEY)
    scores = {}
    # Store more meta data so we can click through and see more of the anlaysis (e.g. which terms appeared in titles, frequency, cooccurance, which titles we were working with, etc)

    print 'Querying Bing...'
    for i, key in enumerate(terms_dict):
        terms = terms_dict[key]
        topic_terms = []
        for trm in terms:
            if destem:
                trm_title = (dbase.get_prestem(trm[1])[0][0]) # TODO what if the stemmed term doesn't exist for some reason?
            else:
                trm_title = trm[0]
            topic_terms.append(trm_title)
        topic_terms.sort() # sort for linear overlapping scans with search titles 
        search_qry = ' '.join(topic_terms)
        topic_terms = map(stem, topic_terms) # TODO make stemming optional on match?
        print '-topic %i of %i: %s' % (i, len(terms_dict), search_qry),

        tmatches = 0
        for j in xrange(0,numtitles, 50):
            try:
                json_response = bing.search(qry=search_qry, top=50, skip=j)
            except HTTPError:
                print 'Error accessing Bing -- make sure your API key is correct' # TODO propagate this message to the display
                return {}
            responses = json_response['d']['results']
            title_terms = map(lambda resp: resp['Title'].strip().lower().split(), responses) #TODO make case sensitive if desired  TODO make stemming optional
            title_terms = [item for sublist in title_terms for item in sublist]
            title_terms = map(stem, title_terms) # make list of lists into flat list
            title_terms.sort()
            tle_n = 0
            top_n=0
            # presorting the lists allows linear scans 
            while tle_n < len(title_terms) and top_n < len(topic_terms):
                cval = cmp(title_terms[tle_n], topic_terms[top_n])
                if cval == 0: # match 
                    tmatches += 1
                    tle_n += 1
                elif cval == -1: # title_terms is > topic_terms 
                    tle_n += 1
                else: # topic_terms > title_terms
                    top_n += 1
        print ': %i' % tmatches
        scores[key] = tmatches
    return scores

def kfold_perplexity(request, analyzer, k=5, start=-1, stop=-1, step= -1, param="ntopics", pertest=0.20, current_step=None, current_fold=None):
    """
    Divides the corpus into k folds and computes the perplexity of each fold given the other k-1 folds
    @param request: the django request object
    @param k: divide the corpus into k folds
    """

    # construct the k-fold corpus
    corpus_file = analyzer.get_param('corpusfile')
    corpus_dir = os.path.dirname(corpus_file)
    kf_dir = os.path.join(corpus_dir, 'kf_divide' + str(k))
    trainf_list = [os.path.join(kf_dir, 'train' + str(knum) + '.dat') for knum in range(k)]
    testf_list = [os.path.join(kf_dir, 'test' + str(knum) + '.dat') for knum in range(k)]

    corpus = open(corpus_file).readlines()
    test_wc = [0]*k
    wc_obj = os.path.join(kf_dir,'wcs.obj')
    # if necessary, split up the corpus
    if not os.path.exists(wc_obj):
        if not os.path.exists(kf_dir):
            os.mkdir(kf_dir)
        for knum in xrange(k):
            trainfile = open(trainf_list[knum], 'w')
            testfile = open(testf_list[knum], 'w')
            for i in xrange(len(corpus)):
                testwrite = False
                if k == 1:
                    if random.random() < pertest:
                        testwrite = True
                elif (i+1) % k == knum:
                    testwrite = True

                if testwrite:
                    testfile.write(corpus[i])
                    test_wc[knum] += _sum_words_in_doc_line(corpus[i])
                else:
                    trainfile.write(corpus[i])
            testfile.close()
            trainfile.close()
            with open(wc_obj,'wb') as wc_out:
                pickle.dump(test_wc,wc_out)

    else:
        test_wc = pickle.load(open(wc_obj,'rb'))

    append_data = False
    if current_fold and current_step:
        append_data = True
        ppxs = analyzer.kf_perplexity([trainf_list[current_fold-1]], [testf_list[current_fold-1]], test_wc, start=current_step, stop=current_step+1, step=10, param=param)
    else:
        ppxs = analyzer.kf_perplexity(trainf_list, testf_list, test_wc, start=start, stop=stop, step=step, param=param)

    csv_file = '%s/ppt_%s_%i_%i_%i.csv' % (analyzer.get_param('outdir'), param, start, stop, step)
    _write_data(ppxs, csv_file, append_data)
    #self._write_data(ppts, json_file) # TODO implement intermediate results
    print ppxs
    return ppxs

def _write_data(data, file, do_append):
    write_level = 'w'
    if os.path.exists(file) and do_append:
        write_level = 'a'
    try:
        data_out = open(file, write_level)
        data_out.write(str(data)) # TODO write this as a csv
        data_out.close()
    except:
        print '\n\nWARNING unable to save data into %s' % file

def _sum_words_in_doc_line(dl):
    terms = dl.strip().split()[1:]
    return sum(map(lambda x: int(x.split(":")[1]), terms))


def doc_graph(request, alg_db, alg=''):
    """
    Create a table-like graph of doc: topic distribution
    """
    myrelations = relations(alg_db)
    documents = myrelations.get_docs()

    docs_data = []
    for doc in documents:
        topics = myrelations.get_top_related_topics(doc, -1)
        topic_keys = topics.keys()
        topic_keys.sort(lambda x, y: -cmp(topics[x], topics[y]))

        total_score_sum = 0
        for key in topic_keys:
            total_score_sum += topics[key]

        tops_data = []
        remaining_topics = ''
        for topic in topic_keys:
            rel_pct = (topics[topic] / total_score_sum)*100
            if rel_pct > 0:
                tops_data.append({'rel_pct':rel_pct, 'get_safe_title':topic.get_safe_title(), 'title':topic.title, 'id':topic.id})
            else:
                if remaining_topics == '':
                    remaining_topics = topic.title
                elif len(remaining_topics) < 150:
                    remaining_topics += ', ' + topic.title
                else:
                    break
        if len(remaining_topics) >= 150:
            remaining_topics += '...'
        docs_data.append({'group_data': doc, 'data':tops_data, 'remainder':{'title':remaining_topics}})


    return render_to_response('table-graph.html', {'input':docs_data, 'alg':alg, 'group_data_type': 'documents', 'data_type':'topics'}, context_instance=RequestContext(request))

def term_graph(request, alg_db, alg=''):
    myrelations = relations(alg_db)
    nterm_disp = 100
    terms = myrelations.get_terms(nterm_disp)     # TODO use AJAX to pull out more terms as desired -- sort by frequency

    terms_data = []
    for trm in terms:
        topics = myrelations.get_top_related_topics(trm, -1)
        topic_keys = topics.keys()
        topic_keys.sort(lambda x, y: -cmp(topics[x], topics[y])) # TODO are these sorts necessary since we're now sorting in database?

        remaining_topics = ''
        tops_data = []
        for topic in topic_keys:
            rel_pct = myrelations.get_relative_percent(topic, trm)
            if rel_pct > 0:
                tops_data.append({'rel_pct':rel_pct, 'get_safe_title':topic.get_safe_title(), 'title':topic.title, 'id':topic.id})
            else:
                if remaining_topics == '':
                    remaining_topics = topic.title
                elif len(remaining_topics) < 150:
                    remaining_topics += ', ' + topic.title
                else:
                    break

        if len(remaining_topics) >= 150:
            remaining_topics += '...'
        terms_data.append({'group_data': trm, 'data':tops_data, 'remainder':{'title':remaining_topics}})

    return render_to_response('table-graph.html', {'input':terms_data, 'alg':alg, 'group_data_type': 'terms', 'data_type':'topics'}, context_instance=RequestContext(request))

def term_list(request, alg_db, alg=''):   # TODO: make this more django-like
    myrelations = relations(alg_db)
    terms = myrelations.get_terms()

    terms_table = '<div ><table onmouseover="show_count_bar()" onmouseout="hide_count_bar()">\n'
    for i in range(0, len(terms), 8):
        terms_table += '<tr>\n'
        for j in range(8):
            if (i + j) < len(terms):
                term_count = myrelations.get_term_count(terms[i+j])
                terms_table += '<td title="' + str(int(term_count)) + '" onclick="window.location.href=\'../terms/' + terms[i+j].get_safe_title() + '-' + str(terms[i+j].id) + '\'" onmouseover="count_adjust(' + str(100.0*(term_count - myrelations.term_score_range[0]) /(myrelations.term_score_range[1] - myrelations.term_score_range[0])+1) + ')">' + terms[i+j].title + '</td>\n'
            else:
                terms_table += '<td class="blank"></td>'
        terms_table += '</tr>\n'
    terms_table += '</table></div>\n'
    terms_table = '<div id="list">' + terms_table + '</div>'
    terms_table += '<div onmouseover="show_count_bar()" onmouseout="hide_count_bar()"><table class="hidden"><tr><td class="count"></td><td class="total"></td></tr></table></div>'
    template = 'table-graph.html'
    return render_to_response(template, {'data':terms_table, 'alg':alg}, context_instance=RequestContext(request))

def topic_graph(request, alg_db, alg='', maxterms=100):
    """
    Build a topic-term graph to display the prevalence of each term in each topic
    """
    myrelations = relations(alg_db)
    topics_list = myrelations.get_topics()

    topics_data = []
    for topic in topics_list:
        # extract the topic title and id
        terms = topic.get_terms(maxterms)

        remaining_terms = ''
        topic.get_terms()
        terms_data = []
        for trm in terms:
            rel_pct = round(topic.get_relative_percent(trm),4)
            if rel_pct > 0:
                terms_data.append({'rel_pct':rel_pct, 'get_safe_title':trm.get_safe_title(), 'title':trm.title, 'id':trm.id})
            else:
                if remaining_terms == '':
                    remaining_terms = trm.title
                elif len(remaining_terms) < 150:
                    remaining_terms += ', ' + trm.title
                else:
                    break
        if len(remaining_terms) >= 150:
            remaining_terms += '...'

        topics_data.append({'group_data': topic, 'data':terms_data, 'remainder':{'title':remaining_terms}})

    return render_to_response('table-graph.html', {'input':topics_data, 'alg':alg, 'group_data_type': 'topics', 'data_type':'terms'}, context_instance=RequestContext(request))

def topic_presence_graph(request, alg_db, alg=''):
    """
    Returns a graph of the relative presence of each topic (bar graph)
    """
    myrelations = relations(alg_db)
    topics = myrelations.get_topics()

    top_data=[]
    max_overall_score = myrelations.get_overall_score(topics[0])
    for tpc in topics:
        overall_topic_score = myrelations.get_overall_score(tpc)
        width = overall_topic_score / max_overall_score * 100.0
        top_data.append({'group_data':tpc, 'data':[{'rel_pct':width}]})

    return render_to_response('table-graph.html', {'input':top_data, 'alg':alg, 'group_data_type': 'topics'}, context_instance=RequestContext(request))

# TODO address repeated code, as well as possibly unoptimized database queries
def get_term_page(request, alg_db, term_title, termid, term_cutoff=NUM_TERMS, doc_cutoff=10, topic_cutoff=10, alg=''):
    """
    returns the term page to the user with related terms, documents, and topics
    """
    # init
    myrelations = relations(alg_db)
    term = Term(termid, term_title)

    # related topics
    # pie array
    topics = myrelations.get_top_related_topics(term, topic_cutoff)
    topic_keys = topics.keys()
    topic_keys.sort(lambda x, y: -cmp(topics[x], topics[y]))
    piearray = get_js_term_topics_pie_array(myrelations, term, topic_keys)

    topics = myrelations.get_top_related_topics(term, topic_cutoff)
    topic_keys = topics.keys()
    topic_keys.sort(lambda x, y: -cmp(topics[x], topics[y]))
    leftcol = {'piearray':piearray, 'data': topic_keys[:topic_cutoff], 'webname':'topics'}

    # related docs
    docs = myrelations.get_top_related_docs(term, doc_cutoff)
    doc_keys = docs.keys()
    doc_keys.sort(lambda x, y: -cmp(docs[x], docs[y]))
    midcol = {'data': doc_keys[:doc_cutoff], 'webname':'documents'}

    # related terms
    top_related_terms = myrelations.get_related_terms(term, term_cutoff)
    rightcol = {'data': top_related_terms[:term_cutoff], 'webname':'terms'}



    return render_to_response("three-column-vis.html", {'leftcol':leftcol, 'midcol':midcol, 'rightcol':rightcol, 'title':term.title}, context_instance=RequestContext(request))

def get_topic_page(request, alg_db, topic_title, topicid, term_cutoff=NUM_TERMS, doc_cutoff=10, topic_cutoff=10, alg=''):
    """
    returns the topic page to the user with related terms, documents, and topics
    """

    myrelations = relations(alg_db)
    if not topic_title[0] == '{':
        topic_title = '{' + ', '.join(topic_title.strip().split()) + '}'
    topic = Topic(myrelations, topicid, topic_title)

    # related terms
    terms = topic.get_terms(term_cutoff)
    # add an interactive pie chart
    piearray = get_js_topic_term_pie_array(topic, terms) # TODO replace this in template
    leftcol = {'piearray':piearray, 'data':terms, 'webname':'terms'}


#    # related docs
    docs = myrelations.get_top_related_docs(topic, doc_cutoff)
    doc_keys = docs.keys()
    doc_keys.sort(lambda x, y: -cmp(docs[x], docs[y]))
    midcol = {'data': doc_keys[0:doc_cutoff],  'webname':'documents'}

#    # related topics
    topics = myrelations.get_top_related_topics(topic, topic_cutoff)
    topic_keys = topics.keys()
    topic_keys.sort(lambda x, y: cmp(topics[x], topics[y]))
    rightcol = {'data': topic_keys[0:topic_cutoff], 'webname':'topics'}

    return render_to_response("three-column-vis.html", {'leftcol':leftcol, 'midcol':midcol, 'rightcol':rightcol, 'title':topic.title}, context_instance=RequestContext(request))


def get_doc_page(request, alg_db, doc_title, docid, docloc, doc_cutoff=10, topic_cutoff=10, alg=''):
    """
    return the document page to the user with related terms and topics and the document text
    TODO limit the length of the document returned to first XY bytes
    """

    myrelations = relations(alg_db)
    doc = Document(docid, doc_title)

    topics = myrelations.get_top_related_topics(doc, topic_cutoff)
    piearray = get_js_doc_topic_pie_array(topics)
    # related topics
    topic_keys = topics.keys()
    topic_keys.sort(lambda x, y: -cmp(topics[x], topics[y]))
    leftcol = {'piearray':piearray, 'data':topic_keys[:topic_cutoff], 'webname':'topics'}


    # related documents      
    docs = myrelations.get_top_related_docs(doc, doc_cutoff)
    doc_keys = docs.keys()
    doc_keys.sort(lambda x, y: cmp(docs[x], docs[y]))
    rightcol = {'data':doc_keys[:topic_cutoff], 'webname':'documents'}



    try:
        doc_text_file = open(os.path.join(docloc, slugify(unicode(doc_title))),'r')
    except IOError:
        doc_text_file = open(os.path.join(docloc, slugify(unicode(doc_title)) + '.txt'),'r') # TODO fix hack

    midcol = {'doctext':doc_text_file.read()}
    doc_text_file.close()
    return render_to_response("three-column-vis.html", {'leftcol':leftcol,
        'rightcol':rightcol, 'midcol':midcol, 'title':doc.title}, context_instance=RequestContext(request))

# TODO these three methods have some redundency and could use generators instead of strings
def get_js_doc_topic_pie_array(topics):
    """
    obtain the topic-term data in appropriate data format for flot
    """
    array_string = "["
    topic_keys = topics.keys()
    topic_keys.sort(lambda x, y: -cmp(topics[x], topics[y]))
    key_count = 0
    for key in topic_keys:
        array_string += "{label:\"" + key.get_safe_title() + "\", data:" + str(topics[key]) + "}"
        key_count += 1
        if key_count != len(topic_keys):
            array_string += ", "
    array_string += "]"
    return array_string

def get_js_topic_term_pie_array(topic, terms):
    """
    obtain the topic-term data in appropriate data format for flot
    """
    array_string = "["
    term_count = 0
    for term in terms:
        rel_percent = exp(topic.terms[term])
        array_string += "{label:\"" + term.get_safe_title() + "\", data:" + str(rel_percent) + "}"
        term_count += 1
        if term_count != len(terms):
            array_string += ", "
    array_string += "]"
    return array_string

def get_js_term_topics_pie_array(myrelations, term, topic_keys):
    """
    obtain the term-topic data in appropriate data format for flot
    """
    array_string = "["
    topic_ct = 0
    for topic in topic_keys:
        rel_percent = myrelations.get_relative_percent(topic, term)/100.0
        array_string += "{label:\"" + topic.get_safe_title() + "\", data:" + str(rel_percent) + "}"
        topic_ct += 1
        if topic_ct != len(topic_keys):
            array_string += ", "
    array_string += "]"
    return array_string



             
        
        
                         