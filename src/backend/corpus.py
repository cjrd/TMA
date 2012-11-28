import os
import shutil
import re
import tempfile
from lib.porter2 import stem  
from src.backend.tma_utils import slugify
from src.backend.db import db
from math import log10
import pdb  

class Corpus:  # TODO use tma_utils TextCleaner
    """
    A corpus class for TMA, parses textual data.

    """

    
    def __init__(self, workdir, remove_case=True, stopwordfile=None, usepara=False, dostem=True,
            minwords = 25, parafolder='paradocs', corpus_db='corpusdb.sqlite', make_stem_db=-1):
        """
        @param workdir: directory to save and manipulate corpus data as desired
        @param remove_case: remove the case from terms in the corpus
        @param stopwordfile: file containing the stopwords
        @param usepara: use paragraphs as documents
        @param dostem: stem the terms
        @param minwords: minimum number of words to constitute a document
        @param parafolder: name of paragraph/document folder
        @param corpus_db: name of corpus database
        @param make_stem_db: include a stemming database
        """
        self.workdir = os.path.join(workdir,'corpus')
        os.mkdir(self.workdir)
        self.textdir = os.path.join(self.workdir, 'textdir')
        os.system('mkdir %s' % self.textdir)
        self.remove_case = remove_case
        self.rawtextfiles = []
        self.titles = [] # stores the (inferred) title of each document
        self.docct = 0
        self.vocabct = 0
        self.wordct = 0
        self.vocab = dict()
        self.corpusfile = os.path.join(self.workdir,'corpus.dat')
        self.corpus_used = False
        self.paradir = os.path.join(workdir,parafolder) 
        self.no_title_ct = 0 # keep track of documents with no title 
        self.corpus_db = os.path.join(self.workdir,corpus_db)
        self.parsed_data = False
        self.pdf_list = []

        # keep a db to reverse stem words
        if make_stem_db == -1:
            self.make_stem_db = dostem
        else:
            self.make_stem_db = make_stem_db
        
        self.stopwords = {}
        if stopwordfile:
            try:
                stopfile = open(stopwordfile,'r')
                for sw in stopfile:
                    self.stopwords[sw.strip()] = ''
                stopfile.close()
            except IOError:
                print "WARNING: unable to add the stop word file to the corpus"
        self.notvalidreg = re.compile('\\S*[^\\w\\s]+\\S*') # remove odd words and strange pdftotext conversions (make sure to first remove punctuation)
        
        self.dostem = dostem 
        self.usepara = usepara # should we use paragraphs as documents or entire document?
        self.minwords = minwords # minimum number of valid words to form a paragraph
    
    def add_data(self, dataname, datatype):
        """
        add data of zip, folder, txt or dat
        """
        datatype = datatype.strip().lower()
        if datatype == 'zip':  # TODO add more types of data
            self.parse_zip(dataname)
        elif datatype in ['folder','txt','dat']:
            self.parse_folder(dataname)
    
    def parse_zip(self, dataname):
        """
        parse zip data
        @param dataname: location of the zip file
        """
        unzipdir = tempfile.mkdtemp(dir=self.workdir, suffix='_unzipdir')
        # unzip file into temp dir       
        #pdb.set_trace()
        unzipcmd = 'unzip -d %s %s' % (unzipdir, dataname)
        os.system(unzipcmd)
        self.parse_folder(unzipdir)
        #TODO: delete unzipdir once finished
            
    def parse_folder(self, folder):
        """
        parses the various datatypes in the folder and writes the lda-c format to file
        """
        
        # obtain list of all pdfs (TODO add heterogenous file types)
        pdflist = os.popen("find %s -name '*.pdf' -type f" % folder) 
        pdflist = pdflist.readlines()
        pdflist = map(lambda x: x.strip(), pdflist)
        self.pdf_list.extend(pdflist)
        toparsetexts = []
        if len(pdflist):
            print '--- beginning pdf to text conversion ---'
            for pdf in pdflist:
                doctitle = self._obtain_clean_title(pdf)
                txtname = self.textdir + '/%s.txt' % doctitle
                cmd = 'pdftotext %s %s' % (pdf, txtname) # TODO: figure out and print which documents did not convert
                os.system(cmd)
                toparsetexts.append(txtname)
                self.rawtextfiles.append(txtname)
            print '--- finished pdf to text conversion ---'
                           
        print '---adding text to corpus---'    
        # add textual data
        txtlist = os.popen("find %s -name '*.txt' -type f" % folder)  # add text files included in folder 
        txtlist = map(lambda x: x.strip(), txtlist) 
        for txtf in txtlist:
            doctitle = self._obtain_clean_title(txtf)
            txtname = self.textdir + '/%s.txt' % doctitle 
            try:
                os.system('ln -s %s %s' % (txtf, txtname))
            except IOError:
                print 'Warning: will not include %s, could not parse text file' % txtf 
                continue
                
            toparsetexts.append(txtname)
            self.rawtextfiles.append(txtname) # TODO: fix code repetition with parsing pdfs
            
        # now add all of the new texts to the corpus
        
        cfile = self.open_corpus()
        if self.usepara: # make a directory for each of the individual paragraphs
            if not os.path.exists(self.paradir): 
                os.makedirs(self.paradir)
        else:     # make a link to the textdir with the same name as the individual paragraph directory
            if not os.path.exists(self.paradir):
                os.system('ln -s %s %s' % (self.textdir, self.paradir))

        # initialize the database to keep track of term-doc occurances
        dbase = db(self.corpus_db)
        if not self.parsed_data:
            dbase.add_table('term_doc_pair(id INTEGER PRIMARY KEY, term INTEGER, doc INTEGER)')
            if self.make_stem_db:
                dbase.add_table('termid_to_prestem(id INTEGER PRIMARY KEY, prestem VARCHAR)')
            
        # add the data to the corpus
        for tfile in toparsetexts:
            title = tfile.split('/')[-1].split('.')[0].replace('-',' ')
            wordcounts = dict() 
            prestem_dic = dict() 
            try:
                infile = open(tfile,'r')
            except IOError:
                print 'WARNING: could not find %s, will not include' % tfile
                continue
            useparanum = 1
            totparanum = 1
            for paraline in infile:
                totparanum += 1
                words = paraline.split()
                for wrd in words:
                    wrd = self.parse_word(wrd)
                    if wrd=='':
                        continue
                    else:
                        prestem = wrd 
                        if self.dostem:
                            wrd = stem(wrd)
                        if wordcounts.has_key(wrd):
                            wordcounts[wrd] += 1
                        else:
                            wordcounts[wrd] = 1     
                            # keep track of the unstemmed forms of new words for later reference. TODO this currently keeps the unstemmed form of the  first encounter of a stemmed word: perhaps make more general?
                            if self.make_stem_db and not self.vocab.has_key(wrd): 
                                prestem_dic[wrd] = prestem
                                 
                if self.usepara:
                    if sum(wordcounts.values()) > self.minwords:
                        self.write_doc_line(cfile, wordcounts, dbase, prestem_dic)
                        usetitle = title + ' [P%d]' % useparanum
                        self.titles.append(usetitle)    
                        if not isinstance(usetitle, unicode):
                            usetitle = unicode(usetitle)                               
                        self.write_document(os.path.join(self.paradir, slugify(usetitle)),paraline)
                        useparanum += 1  
                    wordcounts = dict()
                    prestem_dic = dict() 
            infile.close()
            if not self.usepara:
                if sum(wordcounts.values()) > self.minwords: 
                    self.write_doc_line(cfile, wordcounts, dbase, prestem_dic)
                    self.titles.append(title)
        cfile.close()
        dbase.commit()
        if not self.parsed_data:
            dbase.add_index('term_doc_pair_idx1 ON term_doc_pair(term)')
            dbase.add_index('term_doc_pair_idx2 ON term_doc_pair(doc)')
            dbase.commit()
        print '--- finished adding text to corpus ---'
        print
        self.parsed_data = True
        
    def write_doc_line(self, cfile, wordcounts, dbase, prestem_dic = None):
        """
        write a document line in lda-c format
        """
        cfile.write('%d ' % len(wordcounts))
        for wkey in wordcounts.keys():  
            if not self.vocab.has_key(wkey):
                self.vocab[wkey] = self.vocabct
                self.vocabct += 1 
                if self.make_stem_db and prestem_dic:
                    dbase.insert_termid_prestem(self.vocab[wkey], prestem_dic[wkey])
            dbase.insert_term_doc_pair(self.vocab[wkey], self.docct)
            cfile.write('%d:%d ' % (self.vocab[wkey],wordcounts[wkey]))
        cfile.write('\n')
        self.wordct += sum(wordcounts.values())
        self.docct += 1 
        
    def _obtain_clean_title(self, path):
        """
        obtain the slugified title of a document
        """
        splitname = path.split('/')[-1].split('.')
        # remove unwanted filetitles
        splitname = map(slugify, map(unicode,splitname))  
        doctitle = ''.join(splitname[ : len(splitname) - 1])
        if len(doctitle) == 0:
            doctitle = "notitle" + str(self.no_title_ct)
            self.no_title_ct += 1  
        return doctitle

    def tfidf_clean(self, top_k_terms=5000, min_df=5):
        """
        Use tf-idf to clean the corpus.
        Takes the top tf-idf score of each term and retains the top top_k_terms terms
        Warning: by default tfidf_clean changes the corpus's corpusfile to the cleaned version
        and moves the original version to {{original_name}}-pre_tfidf
        @param top_k_terms: keep the top_k_terms terms by tf-idf rank
        @param min_df: minimum document frequency for the terms
        """
        if not self.corpus_used:
            print "WARNING: You must first parse some data before calling tfidf_clean"
            return False
        orig_corpusfile = self.corpusfile + '-pre_tfidf'
        shutil.move(self.corpusfile, orig_corpusfile)

        # first obtain tf-idf scores for all terms
        tf_list = [0]*self.vocabct
        df_list = [0]*self.vocabct
        tfidf_list = [0]*self.vocabct
        for doc in open(orig_corpusfile,'r'):
            cts = doc.strip().split()[1:] #remove the term count
            term_ct_pairs = map(lambda x: x.split(':'), cts)
            doc_len = sum(map(lambda x: int(x[1]), term_ct_pairs))

            for pair in term_ct_pairs:
                trm = int(pair[0])
                tf = float(pair[1])/doc_len
                df_list[trm] += 1
                if tf > tf_list[trm]:
                    tf_list[trm] = tf

        # calculate tf-df scores
        for i in xrange(self.vocabct):
            tfidf_list[i] = tf_list[i]*log10(float(self.docct)/df_list[i])

        # determine the minimum tf-idf score
        srt_tfidf = sorted(tfidf_list, reverse=True)
        if top_k_terms >= len(srt_tfidf):
            top_k_terms = len(srt_tfidf) - 1
            print "warning tf_idf number of terms exceed length of tfidf scores:", len(srt_tfidf)
        min_score = srt_tfidf[top_k_terms]

        # rewrite the corpus to file, only allowing terms whose max(tf-idf) score exceeds the minimum
        old_to_new_dict = dict()
        self.vocabct = 0
        self.wordct = 0
        writefile = open(self.corpusfile,'w');
        for doc in open(orig_corpusfile,'r'):
            writeline = ''
            cts = doc.strip().split()[1:]
            term_ct_pairs = map(lambda x: x.split(':'), cts)
            doc_term_ct = 0
            for tc_pair in term_ct_pairs:
                tid = int(tc_pair[0])
                if tfidf_list[tid] < min_score or df_list[tid] < min_df:
                    continue
                if not old_to_new_dict.has_key(tid):
                    old_to_new_dict[tid] = self.vocabct
                    self.vocabct += 1
                self.wordct += int(tc_pair[1])
                writeline += str(old_to_new_dict[tid]) + ':' + tc_pair[1] + ' '
                doc_term_ct += 1
            writeline = str(doc_term_ct) + " " + writeline
            writefile.write(writeline + '\n')
        writefile.close()
        remove_ct = len(tfidf_list)-len(old_to_new_dict)
        print 'Processing removed %i of %i terms, keeping %i terms. Min TF-IDF score is: %0.4f' % (remove_ct, len(tfidf_list), len(old_to_new_dict), min_score)

        # update the appropriate databases TODO: perhaps wait to form the databases for efficiecy
        dbase = db(self.corpus_db)
        if self.make_stem_db:
            dbase = db(self.corpus_db)
            oldid_to_prestem = dbase.fetch('SELECT * FROM termid_to_prestem')
            dbase.execute('DROP TABLE termid_to_prestem')
            dbase.add_table('termid_to_prestem(id INTEGER PRIMARY KEY, prestem VARCHAR)')
            id_prestem_list = []
            for op_item in map(list, oldid_to_prestem):
                if old_to_new_dict.has_key(op_item[0]):
                    op_item[0] = old_to_new_dict[op_item[0]]
                    id_prestem_list.append(op_item)
            dbase.executemany('INSERT INTO termid_to_prestem(id, prestem) VALUES(?,?)',id_prestem_list)


        dbase.execute('SELECT * FROM term_doc_pair')
        term_doc_items = []
        for item in dbase.cur:
            if old_to_new_dict.has_key(item[1]):
                item = list(item)
                item[1] = old_to_new_dict[item[1]]
                term_doc_items.append(item[1:])
        dbase.execute('DROP TABLE term_doc_pair')
        dbase.add_table('term_doc_pair(id INTEGER PRIMARY KEY, term INTEGER, doc INTEGER)')
        dbase.executemany('INSERT INTO term_doc_pair(term, doc) VALUES(?,?)', term_doc_items)
        dbase.add_index('term_doc_pair_idx1 ON term_doc_pair(term)')
        dbase.add_index('term_doc_pair_idx2 ON term_doc_pair(doc)') 
        del(dbase)
        del(term_doc_items)


        # update corpus vocab
        oldid_to_term = dict((v,k) for k, v in self.vocab.iteritems())
        self.vocab = {}
        for k,v in old_to_new_dict.iteritems():
            self.vocab[oldid_to_term[k]] = v

    def write_document(self, loc, text):
        """
        write the 'text' of a document to 'loc'
        """
        ofile = open(loc, 'w')
        ofile.write(text)
        ofile.close()

    def write_titles(self, outfile):
        tfile = open(outfile,'w')                                                                                        
        for i in xrange(self.docct):
            tfile.write(self.titles[i] + '\n')
        tfile.close() 
    
    def write_vocab(self, outfile):
        vfile = open(outfile,'w')
        for wrd in sorted(self.vocab, key=self.vocab.get):
            vfile.write('%s\n' % wrd)
        vfile.close()  
        
    def parse_word(self, wrd):
        retword = re.sub('[-\\d.,?;:\'")(!`}{\]\[]','',wrd.strip()) # remove punctuation and such
        if self.remove_case:
            retword = retword.lower()
        if self.notvalidreg.match(retword) or self.stopwords.has_key(retword.lower()) or len(retword) < 3: # TODO make minimum word length a param
            retword = '' 
        return retword 
        
    def open_corpus(self):
        if self.corpus_used:
            openmode = 'a'
        else:
            openmode = 'w'
            self.corpus_used = True
        cfile = open(self.corpusfile, openmode)
        return cfile

    def get_corpus_file(self):
        return self.corpusfile              

    def get_vocab(self):
        return self.vocab
        
    def get_work_dir(self):
        return self.workdir
        
    def get_doc_ct(self):
        return self.docct                       
                                 
    def get_vocab_ct(self):
        return self.vocabct

    def get_word_ct(self):
        return self.wordct

    def setattr(self, attr, value):
        setattr(self, attr, value)

    def clean_pdfs(self):
        """
        remove the pdfs from the given directory
        if no directory is provided try to remove the pdfs specificed in parse_folder
        """
        if self.pdf_list:
            for pdf in self.pdf_list:
                os.remove(pdf)