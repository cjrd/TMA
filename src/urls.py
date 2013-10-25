from django.conf.urls.defaults import *
from src.views import *   
# handle static files locally
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.views.decorators.cache import cache_page


urlpatterns = patterns('',
    (r'^(?i)tma/$', process_form),
    (r'^tma/([^/]+)/([a-zA-Z0-9]+)/([^/]+)?/([^/]+)?/?$', res_disp),# cache_page(60*30)(res_disp))# (add this to cache)
)

urlpatterns += staticfiles_urlpatterns() 
