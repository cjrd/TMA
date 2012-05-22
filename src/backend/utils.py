import string

verbose = False

def set_verbose(value):
    global verbose
    verbose = value

def printv(string):
    if verbose:
        print string

#this allows for static functions in a class, as used in the messages class below
class Callable:
    def __init__(self, anycallable):
        self.__call__ = anycallable

class messages:
    def print_usage():
        print 'Usage: python tmve.py [FILE]'
        print 'Try tmve --help for more information.'
    print_usage = Callable(print_usage)

    def print_help():
        print 'TMVE (Topic Model Visualization Engine) takes topic model information from a database and generates a browing interface in HTML.'
        print '\n    Usage: python tmve.py [FILE]'
        print '\nIn the above example, the FILE should be a plain text file containing information about the browsing template and any information that template would need (i.e. a database filename).'
        print '\nAdditionally, The following options are available:'
        print '    --verbose, -v:   prints messages during browser generation'
        print '                     use: python tmve.py -v [FILE]'
        print '    --help, -h:      prints this help message'
        print '\nPlease read the README file for more information.' #TODO: insert link to online help?
    print_help = Callable(print_help)

    def print_unknown_option(opt):
        print 'tmve: invalid option ' + opt
    print_unknown_option = Callable(print_unknown_option)

    def print_file_read_error(filename, strerror):
        print 'Error reading file \'' + filename + '\': ' + strerror
    print_file_read_error = Callable(print_file_read_error)
    
    def print_malformed_file(filetype, filename = '', expectation = '', linenum = 0, line = ''):
        if filename == '':
            print filetype + ' file is malformed.'
        else:
            print filetype + ' file \'' + filename + '\' is malformed.'

        if expectation != '':
            	print 'Expected ' + expectation + ' on line ' + str(linenum) + ':'
            	print line
    print_malformed_file = Callable(print_malformed_file)

    def print_error(error):
        print 'Error: ' + error
    print_error = Callable(print_error)

    def print_warning(warning):
        print 'Warning: ' + warning
    print_warning = Callable(print_warning)

# constant strings
COMMENT_BEGIN = '#'
DATABASE_PREFIX = 'database:'
TEMPLATE_PREFIX = 'template:'
HTML_STRINGS_PREFIX = 'html-strings:'
HTML_INSERTS_PREFIX = 'html-inserts:'
