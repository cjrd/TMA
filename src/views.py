from django.core.context_processors import csrf
from django.http import HttpResponseRedirect
from settings import WORKDIR, DATA_DIR, DEFAULT_STOP_WORDS, MAX_WWW_DL_SIZE, MAX_WWW_FS, ALG_LOCS, REMOVE_DWNLD, MAX_NUM_TERMS
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.utils import simplejson

from src.backend.uploadhandler import FSUploadHandler
from src.backend.corpus import Corpus
from src.backend.LDAAnalyzer import LDAAnalyzer
from src.backend.HDPAnalyzer import HDPAnalyzer 
from src.backend.CTMAnalyzer import CTMAnalyzer
from src.backend.tbrowser import *
from src.backend.forms import AnalysisForm, PerplexityForm
from src.backend.DataCollector import DataCollector
from django.core.files.uploadhandler import StopUpload
from src import settings
from src.backend.relations import Term
import pdb
from time import time
import tempfile
import os
import cPickle as pickle


@csrf_exempt # TODO pass a csrf like the perplexity form
def process_form(request):
    request.upload_handlers.insert(0, FSUploadHandler())
    return _process_form(request)

@csrf_protect
def _process_form(request):
    # file parameters TODO: set up relative path directories for distribution and on server
    """
    Process TMA submission form and return errors/problems to user
    """
    notifs = []
    form = None
#    c.update(csrf(request))
    form_errors = False
    if request.method == 'POST':         
        wbasedir = WORKDIR
        workdir = tempfile.mkdtemp(dir = wbasedir, suffix = '_formdata')

        try:
            form = AnalysisForm(request.POST, request.FILES)
            form_is_valid = form.is_valid()
        except StopUpload: # TODO this is not working as expected
            error_message = "Uploaded data must be < %0.1f Mb" % (settings.MAX_UPLOAD_SIZE)
            print error_message
            notifs.append(error_message)
            form_is_valid = False

        if form_is_valid:
            is_valid = False
            has_web_data = False
            has_upload_data = False
            has_arxiv_data = False
            algotype = form.cleaned_data['std_algo']
            doHDP = algotype == 'hdp'
            doLDA = algotype == 'lda'
            doCTM = algotype == 'ctm'
            datatype = form.cleaned_data['toy_selected'] # what type of data to process?
            
            # handle pdf collection from a given website TODO: put limitations and security checks, etc
            website = form.cleaned_data['url_website']
            if datatype=="#data_url" and len(website) > 0:
                webdir = tempfile.mkdtemp(dir=workdir, suffix='_webdata')
                data_collector = DataCollector(webdir, MAX_WWW_DL_SIZE)
                wwwres = data_collector.collect_www_data(website, MAX_WWW_FS)
                if wwwres == -12:
                    notifs.append("WWW data collection not allowed by robots.txt")
                else:
                    has_web_data = True
                    is_valid = True
                # TODO: Add checking for appropriate filetypes and return the amount of data processed            
            # handle uploaded files
            upfile = form.cleaned_data['upload_file']
            if upfile is None:
                is_valid = is_valid or False   
            elif datatype=="#data_upload":
                ext = os.path.splitext(upfile.name)[1]
                upload_data_name = workdir + '/outdata' + ext
                outfile = open(upload_data_name, 'wb+')
                for chunk in upfile.chunks():
                    outfile.write(chunk)
                outfile.close()
                has_upload_data = True
                is_valid = True

            # handle arXiv files
            arxiv_authors = form.cleaned_data['arxiv_author']
            arxiv_cats = request.POST.getlist('arxiv_subject')
            if arxiv_authors and datatype=="#data_arxiv":
                arxiv_dir = tempfile.mkdtemp(dir=workdir, suffix='_arxivdata')
                data_collector = DataCollector(arxiv_dir, MAX_WWW_DL_SIZE)
                data_collector.collect_arxiv_data(arxiv_authors, arxiv_cats)
                is_valid = True
                has_arxiv_data = True
            else:
                is_valid = is_valid or False

            # use example data
            sample_data_name = ''
            if datatype=="#data_toy":
                sample_data_name = form.cleaned_data['toy_data']
                is_valid = True
           #
           ### Now run the appropriate topic model on the appropriate data and return the TMA ###
           #
            if is_valid:
            # make sure the terms from the previous analyzer are not in main memory
                Term.all_terms = {}
                # parse processing options
                unioptions = form.cleaned_data['process_unioptions']
                doStem = 'stem' in unioptions
                sw_file = None
                if 'remove_stop' in unioptions:
                    sw_file=DEFAULT_STOP_WORDS
                remove_case =  'remove_case' in unioptions

                minwords = form.cleaned_data['process_minwords']
                if not minwords > 0:
                    minwords = 0

                corpus = Corpus(workdir, stopwordfile=sw_file, remove_case = remove_case, dostem = doStem, minwords=minwords)
                if has_upload_data:
                    corpus.setattr('usepara', form.cleaned_data['upload_dockind'] == 'paras')
                    corpus.add_data(upload_data_name, ext[1:]) # TODO return to user if it is not a txt or dat or zip file
                if has_web_data:
                    corpus.setattr('usepara', form.cleaned_data['url_dockind'] == 'paras')
                    corpus.add_data(webdir, 'folder')

                if has_arxiv_data:
                    corpus.setattr('usepara', form.cleaned_data['url_dockind'] == 'paras')
                    corpus.add_data(arxiv_dir, 'folder')
                if sample_data_name:
                    corpus.setattr('usepara', False)
                    corpus.add_data(os.path.join(DATA_DIR, sample_data_name),'folder')

                # remove downloaded pdfs if desired (save some space)
                if REMOVE_DWNLD:
                    corpus.clean_pdfs()

                # decide whether to do tfidf cleaning and min-doc-freq removal
                tfidf_cleanf = form.cleaned_data['process_tfidf']
                if not (tfidf_cleanf > 0 and tfidf_cleanf < 1.0):
                    tfidf_cleanf = 1.0

                min_df = form.cleaned_data['process_min_df']
                if min_df > corpus.get_doc_ct():
                    min_df = corpus.get_doc_ct()

                # do tfidf cleaning if necessary
                tfidf_keep = min(MAX_NUM_TERMS, int(corpus.get_vocab_ct()*tfidf_cleanf))
                if tfidf_keep < corpus.get_vocab_ct() or min_df > 0 :
                    st = time()
                    corpus.tfidf_clean(tfidf_keep, min_df=min_df)
                    print 'Data cleaning took %0.2fs' % (time()-st)

                corpusdir = corpus.get_work_dir()
                corpusfile = corpus.get_corpus_file()
                tmoutdir = workdir + '/' + algotype
                vocabfile = os.path.join(corpusdir,'vocab.txt')
                titlesfile = os.path.join(corpusdir,'titles.txt')
                corpus.write_vocab(vocabfile)
                corpus.write_titles(titlesfile)

                # TODO: make it possible to run multiple algos in one run & compare them :) -- perhaps using an uploaded settings file (too much computation for online version?)
                # set the algortihm params
                gblparams =  {'corpusfile':corpusfile,'vocabfile':vocabfile, 'titlesfile':titlesfile, 'outdir':tmoutdir, 'wordct':corpus.get_word_ct()}
                if doLDA:
                    alphaval = form.cleaned_data['lda_alpha']
                    if not alphaval:
                        alphaval = form.cleaned_data['std_ntopics']/50.0
                    ldaparams = {
                        'type':'est',
                        'alphaval':alphaval,
                        'ntopics':form.cleaned_data['std_ntopics'],
                        'alpha':form.cleaned_data['lda_alpha_tech'],
                        'init':form.cleaned_data['lda_topic_init'],
                        'ldadir': ALG_LOCS['lda'],
                        'var_max_iter':form.cleaned_data['lda_var_max_iter'],
                        'var_convergence':form.cleaned_data['lda_var_conv_thresh'],
                        'em_max_iter':form.cleaned_data['lda_em_max_iter'],
                        'em_convergence':form.cleaned_data['lda_em_conv_thresh']
                        }
                    ldaparams = dict(gblparams.items() + ldaparams.items())
                    analyzer = LDAAnalyzer(ldaparams)
                elif doHDP:
                    hdpparams = {
                        'algorithm':'train',
                        'max_iter':form.cleaned_data['hdp_max_iters'],
                        'init_topics':form.cleaned_data['hdp_init_ntopics'],
                        'gamma_a':form.cleaned_data['hdp_gamma_a'],
                        'gamma_b':form.cleaned_data['hdp_gamma_b'],
                        'alpha_a':form.cleaned_data['hdp_alpha_a'],
                        'alpha_b':form.cleaned_data['hdp_alpha_b'],
                        'sample_hyper':form.cleaned_data['hdp_sample_hyper'],
                        'eta':form.cleaned_data['hdp_eta'],
                        'split_merge':form.cleaned_data['hdp_split_merge'],
                        'restrict_scan': 5,
                        'hdpdir': ALG_LOCS['hdp'],
                        'save_lag':-1,
                        'ndocs':corpus.get_doc_ct()
                        }
                    hdpparams = dict(gblparams.items() + hdpparams.items())
                    analyzer = HDPAnalyzer(hdpparams)
                elif doCTM:
                    ctmparams = {'type':'est',
                        'ntopics':form.cleaned_data['std_ntopics'],
                        'init':form.cleaned_data['ctm_topic_init'],
                        'ctmdir': ALG_LOCS['ctm'],
                        'var_max_iter':form.cleaned_data['ctm_var_max_iter'],
                        'var_convergence':form.cleaned_data['ctm_var_conv_thresh'],
                        'em_max_iter':form.cleaned_data['ctm_em_max_iter'],
                        'em_convergence':form.cleaned_data['ctm_em_conv_thresh'],
                        'cg_max_iter':-1,
                        'cg_convergence':1e-6,
                        'lag':50,
                        'covariance_estimate':form.cleaned_data['ctm_cov_tech'],
                        'nterms':corpus.get_vocab_ct(),
                        'ndocs': corpus.get_doc_ct()}
                    ctmparams = dict(gblparams.items() + ctmparams.items())
                    analyzer = CTMAnalyzer(ctmparams)

                    print analyzer.get_params()

                analyzer.do_analysis()
                # save the analyzer for later use
                pickle.dump(analyzer,open(os.path.join(analyzer.get_param('outdir'), 'analyzer.obj'),'w'))
                analyzer.create_relations()
                analyzer.createJSLikeData()
                return HttpResponseRedirect('/tma/' +  '_'.join(workdir.split('/')[-1].split('_')[0:-1]) + '/' + analyzer.params['alg'] + '/topic-list')
            else:
                notifs.append('Unable to process data.')
                notifs.append('See <a target="_blank" href="http://github.com/cjrd/TMA/wiki/TMA-Interface">the TMA interface documentation</a> for assistance.')
        else:
            form_errors = True
    if not form: # data was not actually valid (TODO perhaps do a try/catch and wind up here...)
        form = AnalysisForm()
    if form_errors:
        notifs.insert(0,'There were errors, see below')
        
    retpage = render_to_response('init-page.html', {'form': form, 'notifications':notifs}, context_instance=RequestContext(request))
    return retpage

def perplexity_form(request, algloc):
    perplex_res = None
    if request.GET:
        form = PerplexityForm(request.GET)
        if form.is_valid():
            nfolds = form.cleaned_data['folds']
            start = form.cleaned_data['start']
            stop = form.cleaned_data['stop']
            step = form.cleaned_data['step']
            current_step = form.cleaned_data['current_step']
            current_fold = form.cleaned_data['current_fold']
            perplex_res = kfold_perplexity(request, pickle.load(open(os.path.join(algloc, 'analyzer.obj'))), param='ntopics',
                                           k=nfolds, start=start, stop=stop, step=step, current_step=current_step, current_fold=current_fold)
    else:
        form = PerplexityForm()
    if perplex_res:
        return HttpResponse(simplejson.dumps(perplex_res), mimetype='application/javascript')
    else:
        return render_to_response('perplexity-form.html', {'form':form}, context_instance=RequestContext(request))

def res_disp(request, folder, alg, res, param = ''):  

    alg = alg.lower()
    
    # TODO implement better error catching 
    # pdb.set_trace()       
    dataloc = os.path.join(WORKDIR, folder + '_formdata');
    algloc = os.path.join(dataloc, alg)
    tma_dbase = os.path.join(algloc, 'tma.sqlite')
    alg_db = db(tma_dbase)

    corpus_dbloc = os.path.join(dataloc, 'corpus', 'corpusdb.sqlite') 
    toplist_numcolumns = 3  
    # TODO: Add other params here
    tst = time()   
    if not param is None:
        prm_id = param.strip().split('-')[-1]  
        prm_text = ' '.join(param.strip().split('-')[0:-1])
    
    if res == 'topic-list' or res =='summary': #TODO, make this one or the other
        output = get_summary_page(request, alg_db, numcolumns=toplist_numcolumns, alg=alg)  
    elif res == 'doc-graph':
        output = table_graph_rel(request, res, alg_db, alg=alg, RPP=99)
    elif res == 'term-graph':
        output = table_graph_rel(request, res, alg_db, alg=alg, RPP=49)
    elif res == 'topic-graph':
        output = table_graph_rel(request, res, alg_db, alg=alg, RPP=29)
    elif res == 'term-list':                       
        output = presence_graph(request, 'terms', alg_db, alg=alg, RPP=199)
    elif res == 'topic-presence':
        output = presence_graph(request, 'topics', alg_db, alg=alg, RPP=99)
    elif res == 'terms':                                    
        output = get_term_page(request, alg_db, prm_text, int(prm_id), alg=alg) 
    elif res == 'topics':
        output = get_topic_page(request, alg_db, prm_text, int(prm_id), term_cutoff=10, doc_cutoff=10, alg=alg);
    elif res == 'documents':
        output = get_doc_page(request, alg_db, prm_text, int(prm_id), os.path.join(dataloc,'paradocs'), topic_cutoff=10, doc_cutoff=10, alg=alg);
    elif res == "model":
        bing_coh_dict_loc = os.path.join(algloc,"bing_coherence_dict.obj")
        bing_coh_dict = {}
        if os.path.exists(bing_coh_dict_loc):
            bing_coh_dict = pickle.load(open(bing_coh_dict_loc,'rb'))
        output = get_model_page(request, alg_db, corpus_dbloc, dataloc, alg=alg, bing_dict=bing_coh_dict, form = PerplexityForm )
    elif res == "relationships":
        output = get_rel_page(request, alg_db, dataloc, alg=alg)
    elif res == "doc-text":
        output = get_doc_text(os.path.join(dataloc,'paradocs'), param)
    elif res == "bing-coherence":
        bing_coh_dict_loc = os.path.join(algloc,"bing_coherence_dict.obj")
        bing_coh_dict = {}
        if os.path.exists(bing_coh_dict_loc):
            bing_coh_dict = pickle.load(open(bing_coh_dict_loc,'rb'))
        if bing_coh_dict =={}:
            topic_terms_dic = pickle.load( open(os.path.join(algloc,TOP_TOPIC_OBJ)) ) # TODO handle the case if the topic term dict is not present
            bing_coh_dict = get_bing_coherence_dict(topic_terms_dic, corpus_dbloc, numtitles = 50)
            pickle.dump(bing_coh_dict, open(bing_coh_dict_loc,'wb'))
        # now pass back the JSON data of the bing_coh_dict
        output = HttpResponse(simplejson.dumps(bing_coh_dict), mimetype='application/javascript')

    elif res == "perplexity":
        output = perplexity_form(request, algloc)
        #perplex_res = kfold_perplexity(request, pickle.load(open(os.path.join(algloc, 'analyzer.obj'))))
        #output = HttpResponse(simplejson.dumps(perplex_res), mimetype='application/javascript')

    else:
        output = HttpResponse('<html><body><strong>Page Not Found (TODO: implement a better "page not found" page)</strong></html></body>')
    del alg_db
    print 'took %f seconds' % (time()-tst)
    
    return output
    
    
   

