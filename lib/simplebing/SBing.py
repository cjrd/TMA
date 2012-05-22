import urllib2, urllib, json

class SBing():
  API = 'http://api.bing.net/json.aspx'
  
  def __init__(self, app_id):
    self.app_id = app_id
    
  def search(self, **params):
    data = {}
    [data.update({key.replace('_','.'):value}) for key,value in params.iteritems()]

    f = urllib2.urlopen('%s?Appid=%s&%s' % (self.API, self.app_id, urllib.urlencode(data)))
    return json.loads(f.read())