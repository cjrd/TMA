import re 
import lib.porter2  
import pdb

def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    import unicodedata 
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    value = unicode(re.sub('[^\w\s-]', '', value).strip().lower())
    return re.sub('[-\s]+', '-', value)    
    
def remove_non_ascii(txt):
    return "".join(i for i in txt if ord(i)<128)  
    
def ids_to_key(id1, id2):
    if type(id1) != int:
        id1 = int(id1)
    if type(id2) != int:
        id2 = int(id2)
    ids = sorted([id1,id2])
    return ids[0], ids[1]

def gen_clean_text(file_obj):
    for line in file_obj:
        yield remove_non_ascii(line)

def mean(alist):
    return float(sum(alist))/len(alist)

def median(alist):
    if len(alist) == 1:
        return alist[0]
    alist = sorted(alist)
    n = len(alist)  
    if n % 2 == 1:
       median = alist[(n+1)/2 - 1]
    else:
        loc1 = n/2 - 1;
        loc2 = n/2;
        median = float((alist[loc1] + alist[loc2]))/2
    return median
        
    
class TextCleaner:
    def __init__(self, not_valid_reg=re.compile('\\S*[^\\w\\s]+\\S*'), remove_chars='[-\\d.,?;:\'")(!`}{\]\[]', stem = lib.porter2.stem, stopword_file=None):        

        self.not_valid_reg = not_valid_reg  
        
        if remove_chars and not remove_chars == '':
            self.remove_chars_re = re.compile(remove_chars) 
        else:
            self.remove_chars_re = None

        self.stem = stem
        
        if stopword_file: 
            self.stopwords = {}
            stopfile = open(stopword_file,'r') 
            for sw in stopfile:
                self.stopwords[sw.strip()] = 0
            stopfile.close()
        else: 
            self.stopwords = None
    
    def parse_text(self, text):
        clean_text = [] 
        for word in text.strip().split():
            word = self.parse_word(word)   
            if not word == '':
                clean_text.append(word)
        return clean_text 
        
    def parse_word(self, word):
        if self.remove_chars_re:
            retword = re.sub(self.remove_chars_re,'',word.strip()).lower() # remove punctuation and such
        
        if self.not_valid_reg:
            if self.not_valid_reg.match(retword):
                return ''
            
        if self.stopwords:
            if self.stopwords.has_key(retword):  
                return '' 
        if self.stem:
            retword = self.stem(retword)
                
        return retword
