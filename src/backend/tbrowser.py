# a class to control the djangoiation of the tbrowser
import shutil
from django.shortcuts import render_to_response
from django.template import Context, RequestContext  
from django.http import HttpResponse
from django.utils.html import escape
from settings import BING_API_KEY, WIKI_COCC_DB, WIKI_NUM_ABST
from urllib2 import HTTPError

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
    myrelations = relations(alg_db)
    topics= myrelations.get_topics()

    # HACK: now place into groups of numcolumns for the table TODO find a better way to do this -- use divs instead of tables
    colct = min(len(topics), numcolumns) # number of columns for the data table
    table_topics = []
    ct = 0
    title_line = []

    # restructure the data for easy table filling in the template file TODO restructure to use CSS and avoid this mess!
    for i in xrange(len(topics)):
        topics[i].get_terms(numterms) # get the top numterms terms (prep)

        title_line.append({'title':topics[i].title, 'id':topics[i].topic_id})
        ct += 1
        if ct % colct == 0:
            terms_line = []
            for j in xrange(numterms):
                line = []
                for k in xrange(colct):
                    term_title = topics[ct-colct+k].get_term(j).title
                    term_id = topics[ct-colct+k].get_term(j).id
                    line.append({'title':term_title, 'id':term_id})
                terms_line.append(line)
            table_topics.append({'title_line':title_line, 'terms_line':terms_line})
            title_line = []
    # pdb.set_trace()
    template = 'summary.html'
    
    return render_to_response(template, {'table_topics':table_topics, 'alg':alg}, context_instance=RequestContext(request))

def get_model_page(request, alg_db, corpus_dbloc, dataloc, alg='', num_terms=NUM_TERMS, bing_dict={}, form=None): # TODO stem the terms once for bing and wiki calculations
    """
    Obtain/save the appropriate data for the model-description page
    @param request: the django request object
    @param alg_db: the algorithm database from generate_db
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
    
    
# Coherence from Newman, 2011 Automatic (search index with Bing)   
def get_bing_coherence_dict(terms_dict, corpus_dbloc, numtitles=50):#(terms, corpus_dbloc, numterms=NUM_TERMS, numtitles=100):
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
   #sleep(1) # don't overwhelm Bing?   
    return scores

def kfold_perplexity(request, analyzer, k=5, start=-1, stop=-1, step= -1, param="ntopics", pertest=0.20, current_step=None, current_fold=None):
    """
    Divides the corpus into k folds and computes the perplexity of each fold given the other k-1 folds
       @param request: the django request object
       @param k: divide the corpus into k folds
    """
    #load the previous analyzer and do pdb.set_trace()
    #analyzer = pickle.load(open())

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


def doc_graph(request, alg_db, alg=''):   # TODO: make this more django-like and fix alignment issues
    myrelations = relations(alg_db)
    docs = myrelations.get_docs()
    
    docs_table = ""
    topics_table = ""
    
    for doc in docs:
        docs_table += '<table class="doc-graph-table"><tr><td class="dark-hover"><a href="../documents/' + doc.get_safe_title() + '-' + str(doc.id) + '">' + doc.title + '</a></td></tr></table>\n'
        topics_table += '<table class="high-contrast-table"><tr>'
        topics = myrelations.get_related_topics(doc)
        topic_keys = topics.keys()
        topic_keys.sort(lambda x, y: -cmp(topics[x], topics[y]))
        
        total_percent = 0
        remaining_topics = ''
        
        total_score_sum = 0
        for key in topic_keys:
            total_score_sum += topics[key]
        
        for topic in topic_keys:
            per = topics[topic] / total_score_sum
            if (per != 0):
                topics_table += '<td class="clickable" width="' + str(per * 100) + '%" title="' + topic.title + '" onclick="window.location.href=\'../topics/' + topic.get_safe_title() + '-' + str(topic.id) + '\'"></td>\n'
                total_percent += per
            else:
                if remaining_topics == '':
                    remaining_topics = topic.title
                elif len(remaining_topics) < 150:
                    remaining_topics += ', ' + topic.title

        if len(remaining_topics) >= 150:
            remaining_topics += '...'

        if remaining_topics != '' and (100 - total_percent) > 0:
            topics_table += '<td width="' + str((1 - total_percent) * 100) + '%" title="' + remaining_topics + '">&nbsp;</td>\n'
        
        topics_table += '</tr></table>\n'

    doc_graph = '<div id="graph">\n<table width="100.0%">\n<tr><td width="50%">\n\n' + docs_table + '\n</td><td class="bars">\n' + topics_table + '\n</td></tr>\n</table></div>'
    template = 'table-graph.html'
    return render_to_response(template, {'data':doc_graph, 'alg':alg}, context_instance=RequestContext(request))
    
def term_graph(request, alg_db, alg=''):   # TODO: make this more django-like
    myrelations = relations(alg_db)  
    nterm_disp = 500;
    terms = myrelations.get_terms(nterm_disp)     # TODO use AJAX to  pull out more terms as desired
    terms_table = ""
    topics_table = ""  
    
    for term in terms:
        terms_table += '<table class="doc-graph-table"><tr><td class="dark-hover"><a href="../terms/' + term.get_safe_title() + '-' + str(term.id) + '">' + term.title + '</a></td></tr></table>\n'
        topics_table += '<table class="high-contrast-table"><tr>\n'
        topics = myrelations.get_related_topics(term)
        topic_keys = topics.keys()
        topic_keys.sort(lambda x, y: cmp(topics[x], topics[y]))

        total_percent = 0
        remaining_topics = ''

        for topic in topic_keys:
            per = myrelations.get_relative_percent(topic, term)
            if (per != 0):
                topics_table += '<td class="clickable" width="' + str(per * 100) + '%" title="' + topic.title +  '" onclick="window.location.href=\'../topics/' + topic.get_safe_title() + '-' + str(topic.id) + '\'"></td>\n'
                total_percent += per
            else:
                if remaining_topics == '':
                    remaining_topics = topic.title
                elif len(remaining_topics) < 150:
                    remaining_topics += ', ' + topic.title

        if len(remaining_topics) >= 150:
            remaining_topics += '...'

        if remaining_topics != '' and (100 - total_percent) > 0:
            topics_table += '<td width="' + str((1 - total_percent) * 100) + '%" title="' + remaining_topics + '">&nbsp;</td>\n'

        topics_table += '</tr></table>\n' 

    term_graph = '<div id="graph">\n<table width="100.0%">\n<tr><td width="20.0%">\n\n' + terms_table + '\n</td><td class="bars">\n' + topics_table + '\n</td></tr>\n</table></div>'    
    template = 'table-graph.html'
    return render_to_response(template, {'data':term_graph, 'alg':alg}, context_instance=RequestContext(request))

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
               
def topic_graph(request, alg_db, alg=''):   # TODO: make this more django-like and fix alignment issues    
    myrelations = relations(alg_db)  
    topics = myrelations.get_topics() 
    
    topics_table = ""
    terms_table = ""
    
    for topic in topics:                    
        topics_table += '<table class="doc-graph-table" width="250px"><tr><td class="dark-hover" onclick="window.location.href=\'../topics/' + topic.get_safe_title() + '-' + str(topic.id) + '\'">' + topic.title + '</td></tr></table>\n'
        terms_table += '<table width="100%" class="high-contrast-table"><tr>\n'
        terms = myrelations.get_topic_terms(topic)
        term_keys = terms.keys()
        term_keys.sort(lambda x, y: -cmp(terms[x], terms[y]))
        
        total_percent = 0
        remaining_terms = ''
        topic.get_terms()
        for term in term_keys:
            per = topic.get_relative_percent(term)
            if (per != 0):
                terms_table += '<td class="clickable" width="' + str(per*100) + '%" title="' + term.title + '" onclick="window.location.href=\'../terms/' + term.get_safe_title() + '-' + str(term.id) + '\'"></td>\n'
                total_percent += per
            else:
                if remaining_terms == '':
                    remaining_terms = term.title
                elif len(remaining_terms) < 150: #make 150 a const
                    remaining_terms += ', ' + term.title
                else:
                    break

        if len(remaining_terms) >= 150:
            remaining_terms += '...'

        if (100 - total_percent) > 0:
            terms_table += '<td width="' + str((1 - total_percent)*100) + '%" title="' + remaining_terms + '">&nbsp;</td>\n'
        
        terms_table += '</tr></table>\n'
                                                                                                                                                                                  
    topic_graph = '<div id="graph">\n<table width="100.0%">\n<tr><td width="35%">\n\n' + topics_table + '\n</td><td class="bars">\n' + terms_table + '\n</td></tr>\n</table></div>'
    template = 'table-graph.html'
    return render_to_response(template, {'data':topic_graph, 'alg':alg}, context_instance=RequestContext(request))
    
def topic_presence_graph(request, alg_db, alg=''):
    myrelations = relations(alg_db)  
    topics = myrelations.get_topics() 
    
    topic_table = '<table width="100%">\n'
   
    max_overall_score = myrelations.get_overall_score(topics[0])
    for topic in topics:
        overall_topic_score = myrelations.get_overall_score(topic) # colo: this is causing problems  with hdp
        width = overall_topic_score / max_overall_score * 100.0
        topic_table += '<tr><td><table width="' + str(width)  + '%" ><tr><td class="high-contrast" title="' + str(int(overall_topic_score)) + '"onclick="window.location.href=\'../topics/' + topic.get_safe_title() + '-' + str(topic.id) + '\'">' + topic.title + '</td></tr></table></td></tr>\n'
    
    topic_table += '</table>'
    template = 'table-graph.html'       
    topic_table = '<div id="graph">\n' + topic_table   + '</div>' # TODO: fix HDP topic graph output
    return render_to_response(template, {'data':topic_table, 'alg':alg}, context_instance=RequestContext(request))

# TODO address repeated code here, as well as unoptimized database queries: need to fix!  OPTIMIZE
def get_term_page(request, alg_db, term_title, termid, term_cutoff=NUM_TERMS, doc_cutoff=10, topic_cutoff=10, alg=''):
    # init
    myrelations = relations(alg_db)
    term = Term(termid, term_title)   
    
    # related terms
    top_related_terms = myrelations.get_related_terms(term, term_cutoff)             
    terms_column = make_column(top_related_terms, 'terms')     
    
    # related docs
    docs = myrelations.get_related_docs(term)
    doc_keys = docs.keys()       
    doc_keys.sort(lambda x, y: -cmp(docs[x], docs[y]))
    doc_column = make_column(doc_keys[:doc_cutoff], 'documents')

    # related topics
    topics = myrelations.get_top_related_topics(term, topic_cutoff)
    topic_keys = topics.keys()
    topic_keys.sort(lambda x, y: -cmp(topics[x], topics[y])) 
    topic_column = make_column(topic_keys[:topic_cutoff], 'topics')
    
    return render_to_response("three-column-vis.html", {'leftcol':terms_column, 'centercol':doc_column,
        'rightcol':topic_column, 'title':term.get_safe_title(), 'alg':alg}, context_instance=RequestContext(request))  
    
def get_topic_page(request, alg_db, topic_title, topicid, term_cutoff=NUM_TERMS, doc_cutoff=10, topic_cutoff=10, alg=''):  
    """
    returns the topic page to the user with related terms, documents, and topics
    TODO: make this page also display the top N terms of the topic
    """
    
    myrelations = relations(alg_db)
    if not topic_title[0] == '{':
        topic_title = '{' + ', '.join(topic_title.strip().split()) + '}'
    topic = Topic(myrelations, topicid, topic_title)
    # related terms
    terms = topic.get_terms(term_cutoff)
    term_column = ''#make_column(terms, 'terms')  
    term_num = 0
    for trm in terms:
        term_column += '<tr class="list"><td id="' + trm.get_safe_title() + '" onmouseover="highlight(' + str(term_num) + ')" onmouseout="unhighlight()" onclick="window.location.href=\'../terms/' + trm.get_safe_title() + '-' + str(trm.id) + '\'">' + trm.title + '</td></tr>\n'
        term_num += 1                     
    term_column = add_title('related terms',term_column)  
    term_column = add_canvas(term_column) 
    jsarray = get_js_topic_term_pie_array(myrelations, topic, terms)

    # related docs 
    docs = myrelations.get_top_related_docs(topic, doc_cutoff)
    doc_keys = docs.keys()
    doc_keys.sort(lambda x, y: -cmp(docs[x], docs[y]))  
    doc_column =  make_column(doc_keys[0:doc_cutoff],'documents')
                                
    # related topics
    topics = myrelations.get_related_topics(topic)
    topic_keys = topics.keys()
    topic_keys.sort(lambda x, y: cmp(topics[x], topics[y]))
    topic_column = make_column(topic_keys[0:topic_cutoff], 'topics')

    return render_to_response("three-column-vis.html", {'leftcol':term_column, 'centercol':doc_column,
        'rightcol':topic_column, 'title':topic.title, 'jsarray':jsarray}, context_instance=RequestContext(request))    
        
 
def get_doc_page(request, alg_db, doc_title, docid, docloc, doc_cutoff=10, topic_cutoff=10, alg=''):
    # init   
    
    myrelations = relations(alg_db)         
    doc = Document(docid, doc_title)   
                 
    # related topics
    topics = myrelations.get_top_related_topics(doc, topic_cutoff)
    topic_keys = topics.keys()  
    topic_keys.sort(lambda x, y: -cmp(topics[x], topics[y]))   
    topic_keys = topic_keys[:topic_cutoff]
    topic_column = ''   
    topic_num = 0
    for top in topic_keys:
           topic_column += '<tr class="list"><td id="' + top.get_safe_title() + '" onmouseover="highlight(' + str(topic_num) + ')" onmouseout="unhighlight()" onclick="window.location.href=\'../topics/' + top.get_safe_title() + '-' + str(top.id) + '\'">' + top.title + '</td></tr>\n'
           topic_num += 1  
    topic_column = add_title('related topics', topic_column) 
    topic_column = add_canvas(topic_column)     
    
    # related documents      
    docs = myrelations.get_related_docs(doc)
    doc_keys = docs.keys()
    doc_keys.sort(lambda x, y: cmp(docs[x], docs[y]))
    doc_column = make_column(doc_keys[0:doc_cutoff], 'documents')
    jsarray = get_js_doc_topic_pie_array(myrelations, doc)

    try:
        doc_text_file = open(os.path.join(docloc, slugify(unicode(doc_title))),'r')
    except IOError:
        doc_text_file = open(os.path.join(docloc, slugify(unicode(doc_title)) + '.txt'),'r') # TODO fix hack

    txt = doc_text_file.read()
    doc_text = '<tr class="doc"><td>\n' + escape(remove_non_ascii(txt)) + '\n</td></tr>' 
    doc_text_file.close()
    return render_to_response("three-column-vis.html", {'leftcol':topic_column, 'centercol':doc_text,
        'rightcol':doc_column, 'title':doc.title, 'jsarray':jsarray, 'alg':alg}, context_instance=RequestContext(request))    
             
def add_title(title, txt):
    return '<tr class="title"> <td> ' + title + ' </td> </tr>\n' + txt  
    
def add_canvas(txt):
    return '<tr>\n<td style="background-color: #FFFFFF; border-left: solid 1px #9C9C9C;border-right: solid 1px #9C9C9C;  border-top: 5px solid #9C9C9C;"><canvas height="200" width="200" id="canvas"></canvas></td>\n</tr>\n' + txt   
    
def make_column(itrable, name):
    column = ''   
    for it in itrable:
        column += '<tr class="list"><td onclick="window.location.href=\'../' + name + '/' + it.get_safe_title() + '-' + str(it.id) + '\'">' + it.title + '</td></tr>\n'
    column = add_title('related ' + name, column) 
    return column
             

def get_js_doc_topic_pie_array(relations, doc):
    array_string = "["

    topics = relations.get_related_topics(doc)
    topic_keys = topics.keys()
    topic_keys.sort(lambda x, y: -cmp(topics[x], topics[y]))
    key_count = 0
    for key in topic_keys:
        array_string += "[" + str(topics[key]) + ", " + "\"../topics/" + key.get_safe_title() + '-' + str(key.topic_id) + "\", \"" + key.get_safe_title() + "\"]"
        key_count += 1
        if key_count != len(topic_keys):
            array_string += ", "

    array_string += "]"

    return array_string

def get_js_topic_term_pie_array(relations, topic, terms):
    array_string = "["
                                         
    term_count = 0
    term_score_total = 0
    for term in terms:
        rel_percent = exp(topic.terms[term])
        array_string += "[" + str(rel_percent) + ", " + "\"../terms/" + term.get_safe_title() + '-' + str(term.id) + "\", \"" + term.get_safe_title() + "\"]"
        term_count += 1
        term_score_total += rel_percent
        if term_count != len(terms):
            array_string += ", "
    if term_score_total < topic.term_score_total:
        array_string += ', [' + str(topic.term_score_total-term_score_total) + ', \"\", \"\"]'

    array_string += "]"

    return array_string

             
        
        
                         