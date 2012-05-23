import urllib2, urllib, json

class SBing():
    def __init__(self, acct_key, API = 'https://api.datamarket.azure.com/Data.ashx/Bing/Search/Web'):
        self.API = API
        self.acct_key = acct_key
        password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, API, '', acct_key)
        handler = urllib2.HTTPBasicAuthHandler(password_mgr)
        self.opener = urllib2.build_opener(handler)
    
    def search(self, qry, top=50, skip=0):
        qry = '%27' + urllib.quote(qry) + '%27'
        url = '%s?Query=%s&$top=%i&$skip=%i&$format=JSON' % (self.API, qry, top, skip)
        f = self.opener.open(url)
        return json.loads(f.read())