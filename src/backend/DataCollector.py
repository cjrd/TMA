import urllib2, cookielib, re, os
#from urllib import quote_plus
import urlparse
import sys
from lib.bs321.BeautifulSoup import BeautifulSoup
import pdb
from src.backend.tma_utils import slugify
import random
class DataCollector:
    """
    TODO write me
    """

    def __init__(self, data_folder, max_dc=100):
        """
        TODO write me
        set the max data collection at 100 mb
        """
        self.data_folder = data_folder
        self.max_dc = max_dc
        self.tot_dl = 0

    def collect_www_data(self, url, filetype='pdf', size_limit=5):
        """
        TODO check for robots.txt and select the type of file
        """

        cj = cookielib.CookieJar()
        opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))

        req = opener.open(url)
        resp = req.read()
        soup = BeautifulSoup(resp)

        # find the pdfs
        data_urls = soup.findAll(name='a', attrs={'href': re.compile('\.pdf')})
        data_sizes = []
        for data in data_urls:
            dn_url = data['href']
            if 'http' not in dn_url:
                # it's a local url
                dn_url = urlparse.urljoin(url, dn_url)

            try:
                open_url = urllib2.urlopen(dn_url, timeout=8)
                cl = open_url.headers['Content-Length']
                if cl:
                    cl = float(cl) / 1000000
                    data_sizes.append(cl)
                    if cl < size_limit and cl + self.tot_dl < self.max_dc:
                        # download file to dataloc
                        fname = dn_url.split('/')[-1].lower()
                        save_file = os.path.join(self.data_folder, fname)
                        self.tot_dl += cl
                        print '%s, %0.2f Mb' % (fname, cl)
                        self._stream_to_file(open_url, save_file)

                else:
                    continue

            except urllib2.HTTPError:
                print 'Problem accessing %s' % dn_url
                continue

            open_url.close()

        print "total downloaded: %0.2f Mb" % self.tot_dl


    def collect_arxiv_data(self, authors=None, cats=None):
#        pdb.set_trace()
        # TODO handle possible errors in data collection
        # extract params from form
        qry = 'http://export.arxiv.org/api/query?search_query='
        if cats:
            cats = map(lambda x: "cat:" + x, cats)
            if len(cats) > 1:
                cats = '%28' + '+OR+'.join(cats) + '%29'
            else:
                cats = cats[0]
            qry += cats
        if authors:
            authors = authors.lower().split(' or ')
            authors = map(lambda x: '%22' + x.replace(' ', '+') + '%22', authors)
            authors = map(lambda x: "au:" + x, authors)
            authors = '+OR+'.join(authors)
            authors = '%28' + authors.replace(' ','+') + '%29' #quote_plus(authors)
            if cats:
                qry += "+AND+"
            qry += authors

        qry += '&max_results=150' # ONLINE LIMITIATION, remove for standalone or set to 2000
        print qry
        req = urllib2.urlopen(qry, timeout=10)
        soup = BeautifulSoup(req.read())

        titles = soup.findAll('title')
        titles = titles[1:] # skip the query title
        titles = map(lambda x: x.text, titles)
        pdf_links = soup.findAll('link', attrs={'title': 'pdf'})
        pdf_urls = map(lambda x: x['href'], pdf_links)

        print 'downloading: %s, %i' % (authors, len(pdf_urls))
        print titles
        print len(pdf_urls)
        # randomly grab the urls so we don't have all article from one author in online version (i.e. with limitations)
        ct = 0
        for urlnum in random.sample(range(len(pdf_urls)), len(pdf_urls)):
            if self._stream_to_file(urllib2.urlopen(pdf_urls[urlnum], timeout=8), os.path.join(self.data_folder, slugify(titles[urlnum]) + '.pdf')):
                ct += 1
        print '\n$$$$\nAdded %i files from arXiv, total downloaded content at %0.2f Mb\n$$$$\n' % (ct, self.tot_dl)

    def _stream_to_file(self, open_url, save_file):
        with open(save_file, 'wb') as file_writer: # TODO do I want to leave the filename the same as dl fname?
            d_size = float(open_url.headers['Content-Length'])/1000000
            if self.tot_dl + d_size > self.max_dc:
                return False
            print 'Downloading file: %s' % save_file.split('/')[-1]
            for datum in open_url:
                file_writer.write(datum)
            file_writer.close()
            self.tot_dl += d_size
            return True

if __name__ == '__main__':
    #dc = DataCollector('/Users/cradreed/scratch')
    #    dc.collect_www_data('http://highenergy.physics.uiowa.edu/~creed/')
    #dc.collect_arxiv_data("Mary Hall Reno OR Michael Jordan OR Michael I. Jordan")
#    sys.exit(1)
#    url = "http://www.physics.uiowa.edu/personnel/faculty.html"
#    page = urllib2.urlopen(url)
#    soup = BeautifulSoup(page)
#    dds = soup.findAll('dd')
#    for el in dds:
#        fac = re.sub("\\&\S+;","",el.text).split()
#        fac = filter(lambda x: not '.' in x, fac)
#        fac = ' '.join(fac)
#        dc.collect_arxiv_data(fac)
#        break

    import os
    from src.backend.tma_utils import slugify
    dataloc = '/Users/cradreed/Desktop/TM-TestData/testdata_news_fuel_845docs.txt'
    outdir = "/Users/cradreed/Research/TMBrowse/current/data/nyt"
    infile = open(dataloc,'r')
    for ct, line in enumerate(infile):
        outfile = open(os.path.join(outdir,'article_%i.txt' % ct),'w')
        outfile.write(line)
        outfile.close()

