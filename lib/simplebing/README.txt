SimpleBing
__________

SimpleBing is a lightweight python wrapper for the Bing Search API. In order to start using the wrapper, you'll need to register for a Bing Search API developer account at http://www.bing.com/toolbox/bingdeveloper/. Once you have your API key you're all set to go.


Using SimpleBing
________________

In the response you'll see search results about elections from the web, news, and image SourceTypes. You can find a list of SourceTypes in the Bing API documentation. 

from simple_bing import SimpleBing

bing = SimpleBing(<YOUR API KEY>)
json_response = bing.search(query='elections', sources='web news image')


Using SourceType Counts
_______________________

You can also specify how many search results you want to get back by specifying a count for your SourceType. The max count you can specify for a SourceType is 15.

json_response = bing.search(query='elections', sources='web news', web_count=15, news_count=2)


Using SourceType Offsets
________________________

When you get search results, there's always going to be a field named 'total' in the json response. That's the total number of results found for the specified query. The Bing API will only return a max of 15 results per SourceType. So you will need to make more requests with an offset count for the specified SourceType if you want to iterate through all the search results. Here you'll get the next 15 results after the first 15 results.

json_response = bing.search(query='elections', sources='web', web_count=15, web_offset=16)

There are other options you can specify but they may not be supported if you try them. You can check out these extra options in the Enumerations secion in the Bing API documentation.