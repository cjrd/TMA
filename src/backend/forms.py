import pdb
from django.contrib.admin.widgets import FilteredSelectMultiple
from django.forms.forms import BoundField
from django.forms.widgets import  Select
from django import forms

class AnalysisForm(forms.Form):
    std_algo = forms.CharField(required=False, label = 'algorithm',
       widget=Select(
           choices=(
           ('lda','Latent Dirichlet Allocation'),
           ('ctm','Correlated Topic Model'),
           ('hdp','Hierarchical Dirichlet Process')
           )), help_text='Select the desired topic-modeling algorithm -- specific details for the slected algorithm can be found under \'Advanced Options\'')
    std_ntopics = forms.IntegerField(required=False, label='number of topics', initial='10', widget=forms.TextInput(attrs={"id":"numtops"}), help_text='Select the number of topics for parametric topic models (HDP does not require this parameter).')
    std_ntopics.auto_id="ntopics"
    process_tfidf = forms.FloatField(required=False, label='valid tf-idf fraction:', initial='0.7', help_text='Sort the terms in the vocabulary by their <a href="http://en.wikipedia.org/wiki/Tf*idf" target="_blank">TF-IDF </a> scores and keep the top \'tf-idf fraction\' -- this technique removes uninformative terms. Set to the tf-idf fraction > 1.0 to not remove any terms.')
    unichoices = (('stem','stem words'),('remove_case','remove case'), ('remove_stop','remove stop words'))
    process_unioptions = forms.MultipleChoiceField(label='', required=False,  choices=unichoices, widget=forms.CheckboxSelectMultiple(attrs={'checked' : 'checked'}))
    # toy data
    toy_data = forms.CharField(required=False, label = 'dataset',
                               widget=Select(
                                   choices=(
                                       ('tmml','Topic Models Mailing Archive [1887 messages]'),
                                       ('pgm','Coursera PGM Video Transcripts [92 transcripts]'),
                                       ('nsf','NSF Grants [1166 abstracts]'),
                                       ('nyt','New York Times [845 articles]')
                                       )),
                               help_text='Test datasets to evaluate various topic models and parameters. See <a href='' target="_blank">TODO</a> for more information on the datasets.')
    # upload data
    url_website = forms.URLField(label='website', required=False, help_text='A maximum of 50 MB of pdfs will be downloaded from the given URL, e.g. provide a URL with  a collection of research papers such as <a href="http://mlg.eng.cam.ac.uk/pub/" target="_blank">http://mlg.eng.cam.ac.uk/pub/</a>.')
    url_dockind = forms.CharField(required=False, label = 'document representation',
                               widget=Select(
                                   choices=(
                                       ('files','Individual Files'),
                                       ('paras','Paragraphs (Lines)')
                                       )), help_text='Specify whether to treat each individual file as a document or each paragraph/line as a document.')
    upload_file  = forms.FileField(required=False, widget=forms.FileInput(attrs={'size':'13'}), label='upload file', help_text='Upload a text file or zip archive. The zip archive can contain text and/or pdf files.')
    upload_dockind = forms.CharField(required=False, label = 'document representation',
                               widget=Select(
                                   choices=(
                                       ('files','Individual Files'),
                                       ('paras','Paragraphs (Lines)')
                                       )), help_text='Specify whether to treat each individual file as a document or each paragraph/line as a document.')
    arxiv_author = forms.CharField(required=False, label='author',help_text='Search <a href="http://www.arxiv.org">arXiv.org</a> for publications by the specified authors.\
        Separate multiple authors or various spellings using \'OR\', e.g. \'Michael I. Jordan OR Michael Jordan OR David Blei OR David M. Blei\', to search for the publications of the two authors.\
        Restrict the author search to specific fields using the \'subjects\' selector below (hold ctrl to select multiple subjects).')
    arxiv_subject= forms.CharField(required=False, label = 'subjects (default=all)',
       widget=Select(attrs={"multiple":"multiple"},
           choices=(
                ("astro-ph", "Astrophysics"),
                ("cs.AR", "Computer Science - Architecture"),
                ("cs.AI", "Computer Science - Artificial Intelligence"),
                ("cs.CL", "Computer Science - Computation and Language"),
                ("cs.CC", "Computer Science - Computational Complexity"),
                ("cs.CE", "Computer Science - Computational Engineering; Finance; and Science"),
                ("cs.CG", "Computer Science - Computational Geometry"),
                ("cs.GT", "Computer Science - Computer Science and Game Theory"),
                ("cs.CV", "Computer Science - Computer Vision and Pattern Recognition"),
                ("cs.CY", "Computer Science - Computers and Society"),
                ("cs.CR", "Computer Science - Cryptography and Security"),
                ("cs.DS", "Computer Science - Data Structures and Algorithms"),
                ("cs.DB", "Computer Science - Databases"),
                ("cs.DL", "Computer Science - Digital Libraries"),
                ("cs.DM", "Computer Science - Discrete Mathematics"),
                ("cs.DC", "Computer Science - Distributed; Parallel; and Cluster Computing"),
                ("cs.GL", "Computer Science - General Literature"),
                ("cs.GR", "Computer Science - Graphics"),
                ("cs.HC", "Computer Science - Human-Computer Interaction"),
                ("cs.IR", "Computer Science - Information Retrieval"),
                ("cs.IT", "Computer Science - Information Theory"),
                ("cs.LG", "Computer Science - Learning"),
                ("cs.LO", "Computer Science - Logic in Computer Science"),
                ("cs.MS", "Computer Science - Mathematical Software"),
                ("cs.MA", "Computer Science - Multiagent Systems"),
                ("cs.MM", "Computer Science - Multimedia"),
                ("cs.NI", "Computer Science - Networking and Internet Architecture"),
                ("cs.NE", "Computer Science - Neural and Evolutionary Computing"),
                ("cs.NA", "Computer Science - Numerical Analysis"),
                ("cs.OS", "Computer Science - Operating Systems"),
                ("cs.OH", "Computer Science - Other"),
                ("cs.PF", "Computer Science - Performance"),
                ("cs.PL", "Computer Science - Programming Languages"),
                ("cs.RO", "Computer Science - Robotics"),
                ("cs.SE", "Computer Science - Software Engineering"),
                ("cs.SD", "Computer Science - Sound"),
                ("cs.SC", "Computer Science - Symbolic Computation"),
                ("gr-qc", "General Relativity and Quantum Cosmology"),
                ("hep-ex", "High Energy Physics - Experiment"),
                ("hep-lat", "High Energy Physics - Lattice"),
                ("hep-ph", "High Energy Physics - Phenomenology"),
                ("hep-th", "High Energy Physics - Theory"),
                ("math-ph", "Mathematical Physics"),
                ("math.AG", "Mathematics - Algebraic Geometry"),
                ("math.AT", "Mathematics - Algebraic Topology"),
                ("math.AP", "Mathematics - Analysis of PDEs"),
                ("math.CT", "Mathematics - Category Theory"),
                ("math.CA", "Mathematics - Classical Analysis and ODEs"),
                ("math.CO", "Mathematics - Combinatorics"),
                ("math.AC", "Mathematics - Commutative Algebra"),
                ("math.CV", "Mathematics - Complex Variables"),
                ("math.DG", "Mathematics - Differential Geometry"),
                ("math.DS", "Mathematics - Dynamical Systems"),
                ("math.FA", "Mathematics - Functional Analysis"),
                ("math.GM", "Mathematics - General Mathematics"),
                ("math.GN", "Mathematics - General Topology"),
                ("math.GT", "Mathematics - Geometric Topology"),
                ("math.GR", "Mathematics - Group Theory"),
                ("math.HO", "Mathematics - History and Overview"),
                ("math.IT", "Mathematics - Information Theory"),
                ("math.KT", "Mathematics - K-Theory and Homology"),
                ("math.LO", "Mathematics - Logic"),
                ("math.MP", "Mathematics - Mathematical Physics"),
                ("math.MG", "Mathematics - Metric Geometry"),
                ("math.NT", "Mathematics - Number Theory"),
                ("math.NA", "Mathematics - Numerical Analysis"),
                ("math.OA", "Mathematics - Operator Algebras"),
                ("math.OC", "Mathematics - Optimization and Control"),
                ("math.PR", "Mathematics - Probability"),
                ("math.QA", "Mathematics - Quantum Algebra"),
                ("math.RT", "Mathematics - Representation Theory"),
                ("math.RA", "Mathematics - Rings and Algebras"),
                ("math.SP", "Mathematics - Spectral Theory"),
                ("math.ST", "Mathematics - Statistics"),
                ("math.SG", "Mathematics - Symplectic Geometry"),
                ("nlin.AO", "Nonlinear Sciences - Adaptation and Self-Organizing Systems"),
                ("nlin.CG", "Nonlinear Sciences - Cellular Automata and Lattice Gases"),
                ("nlin.CD", "Nonlinear Sciences - Chaotic Dynamics"),
                ("nlin.SI", "Nonlinear Sciences - Exactly Solvable and Integrable Systems"),
                ("nlin.PS", "Nonlinear Sciences - Pattern Formation and Solitons"),
                ("nucl-ex", "Nuclear Experiment"),
                ("nucl-th", "Nuclear Theory"),
                ("physics.acc-ph", "Physics - Accelerator Physics"),
                ("physics.ao-ph", "Physics - Atmospheric and Oceanic Physics"),
                ("physics.atom-ph", "Physics - Atomic Physics"),
                ("physics.atm-clus", "Physics - Atomic and Molecular Clusters"),
                ("physics.bio-ph", "Physics - Biological Physics"),
                ("physics.chem-ph", "Physics - Chemical Physics"),
                ("physics.class-ph", "Physics - Classical Physics"),
                ("physics.comp-ph", "Physics - Computational Physics"),
                ("physics.data-an", "Physics - Data Analysis; Statistics and Probability"),
                ("cond-mat.dis-nn", "Physics - Disordered Systems and Neural Networks"),
                ("physics.flu-dyn", "Physics - Fluid Dynamics"),
                ("physics.gen-ph", "Physics - General Physics"),
                ("physics.geo-ph", "Physics - Geophysics"),
                ("physics.hist-ph", "Physics - History of Physics"),
                ("physics.ins-det", "Physics - Instrumentation and Detectors"),
                ("cond-mat.mtrl-sci", "Physics - Materials Science"),
                ("physics.med-ph", "Physics - Medical Physics"),
                ("cond-mat.mes-hall", "Physics - Mesoscopic Systems and Quantum Hall Effect"),
                ("physics.optics", "Physics - Optics"),
                ("cond-mat.other", "Physics - Other"),
                ("physics.ed-ph", "Physics - Physics Education"),
                ("physics.soc-ph", "Physics - Physics and Society"),
                ("physics.plasm-ph", "Physics - Plasma Physics"),
                ("physics.pop-ph", "Physics - Popular Physics"),
                ("cond-mat.soft", "Physics - Soft Condensed Matter"),
                ("physics.space-ph", "Physics - Space Physics"),
                ("cond-mat.stat-mech", "Physics - Statistical Mechanics"),
                ("cond-mat.str-el", "Physics - Strongly Correlated Electrons"),
                ("cond-mat.supr-con", "Physics - Superconductivity"),
                ("q-bio.BM", "Quantitative Biology - Biomolecules"),
                ("q-bio.CB", "Quantitative Biology - Cell Behavior"),
                ("q-bio.GN", "Quantitative Biology - Genomics"),
                ("q-bio.MN", "Quantitative Biology - Molecular Networks"),
                ("q-bio.NC", "Quantitative Biology - Neurons and Cognition"),
                ("q-bio.OT", "Quantitative Biology - Other"),
                ("q-bio.PE", "Quantitative Biology - Populations and Evolution"),
                ("q-bio.QM", "Quantitative Biology - Quantitative Methods"),
                ("q-bio.SC", "Quantitative Biology - Subcellular Processes"),
                ("q-bio.TO", "Quantitative Biology - Tissues and Organs"),
                ("quant-ph", "Quantum Physics"),
                ("stat.AP", "Statistics - Applications"),
                ("stat.CO", "Statistics - Computation"),
                ("stat.ML", "Statistics - Machine Learning"),
                ("stat.ME", "Statistics - Methodology"),
                ("stat.TH", "Statistics - Theory"))))
    #arxiv_subject = forms.ModelMultipleChoiceField(queryset=MyModel.objects.all(), widget=FilteredSelectMultiple("verbose name", is_stacked=False))

     #LDA EM params
    lda_topic_init = forms.CharField(required=False, label = 'topic initialization',
       widget=Select(
           choices=(
           ('seeded','seeded'),
           ('random','random')
           )),
       help_text="""Describes how the topics will be initialized.
       "random" initializes each topic randomly; "seeded" initializes each topic to a distribution smoothed from a randomly
        chosen document.""")
    lda_alpha = forms.FloatField(required=False, label='alpha', initial='', help_text="[initial] Dirichlet hyperparameter -- default = 50 / (number of topics); note: this heuristic stems from using ~ average number of words per file / number of topics")
    lda_alpha_tech = forms.CharField(required=False, label = 'alpha technique',
       widget=Select(
           choices=(
           ('estimate','estimate'),
           ('fixed','fixed')
           )), help_text="iteratievly 'estimate' the Dirichlet parameter or keep it \'fixed\'")
    lda_var_max_iter = forms.IntegerField(required=False, label='max variational iter', initial='20',
        help_text="""The maximum number of iterations of coordinate ascent variational
     inference for a single document.  A value of -1 indicates "full"
     variational inference, until the variational convergence
     criterion is met.""")
    lda_var_conv_thresh = forms.FloatField(required=False, label='variational convergence threshold', initial='1e-6',
        help_text="""The convergence criteria for variational inference.  Stop if
     (score_old - score) / abs(score_old) is less than this value (or
     after the maximum number of iterations).  Note that the score is
     the lower bound on the likelihood for a particular document.""")
    lda_em_max_iter = forms.IntegerField(required=False, label='maximum variational iterations', initial='100', help_text="The maximum number of iterations of variational EM")
    lda_em_conv_thresh = forms.FloatField(required=False, label='EM convergence threshold', initial='1e-4', help_text="""The convergence criteria for varitional EM.  Stop if (score_old -
     score) / abs(score_old) is less than this value (or after the
     maximum number of iterations).  Note that "score" is the lower
     bound on the likelihood for the whole corpus.""")


    # HDP params
    help_hdp_url = '<a href="http://www.cs.berkeley.edu/~jordan/papers/hdp.pdf" target="_blank"> Hierarchical Dirichlet Process (2005)</a>'
    help_split_merge_url = '<a href="http://arxiv.org/pdf/1201.1657.pdf" target="_blank"> A Split-Merge MCMC Algorithm for the Hierarchical Dirichlet Process (2012)</a>'
    hdp_max_iters = forms.IntegerField(required=False, label='max iterations', initial='500', help_text="The maximum number of Gibbs iterations")
    hdp_init_ntopics = forms.IntegerField(required=False, label='initial # of topics', initial='0', help_text="Set the initial number of topics")
    hdp_sample_hyper = forms.CharField(required=False, label = 'sample hyperparameters',
       widget=Select(
           choices=(
           ('true','true'),
           ('false','false')
           )),
       initial='false', help_text="Sample hyperparameters as described in " + help_hdp_url)
    hdp_gamma_a = forms.FloatField(required=False, label='gamma shape', initial='1.0', help_text="Shape for 1st-level concentration parameter, see " + help_hdp_url)
    hdp_gamma_b = forms.FloatField(required=False, label='gamma scale', initial='1.0', help_text="Scale for 1st-level concentration parameter, see " + help_hdp_url)
    hdp_alpha_a = forms.FloatField(required=False, label='alpha shape', initial='1.0', help_text="Shape for 2nd-level concentration parameter, see " + help_hdp_url)
    hdp_alpha_b = forms.FloatField(required=False, label='alpha scale', initial='1.0', help_text="Scale for 2nd-level concentration parameter, see " + help_hdp_url)
    hdp_eta = forms.FloatField(required=False, label='topic Dirichlet param', initial='0.5', help_text="The topic Dirichlet parameter, see the eta parameter in " + help_hdp_url)
    hdp_split_merge = forms.CharField(required=False, label = 'use split-merge MCMC',
       widget=Select(
           choices=(
           ('true','true'),
           ('false','false')
           )),
       initial='false', help_text="Perform split-merge MCMC inference, see " + help_split_merge_url)

    #CTM params TODO fix code repetition with LDA
    ctm_topic_init = forms.CharField(required=False, label = 'topic init',
       widget=Select(
           choices=(
           ('seed','seeded'),
           ('random','random')
           )),
       help_text="""Describes how the topics will be initialized.
       "random" initializes each topic randomly; "seeded" initializes each topic to a distribution smoothed from a randomly
        chosen document.""")
    ctm_cov_tech = forms.CharField(required=False, label = 'covariance estimate',
       widget=Select(
           choices=(
           ('mle','mle'),
           ('shrinkage','shrinkage')
           )), help_text='Use mle of the covariance matrix or use regression-based \'shrinkage\' approach.')
    ctm_var_max_iter = forms.IntegerField(required=False, label='max variational iterations', initial='20',
        help_text="""The maximum number of iterations of coordinate ascent variational
     inference for a single document.  A value of -1 indicates "full"
     variational inference, until the variational convergence
     criterion is met.""")
    ctm_var_conv_thresh = forms.FloatField(required=False, label='variational convergence threshold', initial='1e-6',
        help_text="""The convergence criteria for variational inference.  Stop if
     (score_old - score) / abs(score_old) is less than this value (or
     after the maximum number of iterations).  Note that the score is
     the lower bound on the likelihood for a particular document""")
    ctm_em_max_iter = forms.IntegerField(required=False, label='max EM iterations', initial='100', help_text="The maximum number of iterations of variational EM")
    ctm_em_conv_thresh = forms.FloatField(required=False, label='EM conv thresh', initial='1e-4', help_text="""The convergence criteria for varitional EM.  Stop if (score_old -
     score) / abs(score_old) is less than this value (or after the
     maximum number of iterations).  Note that "score" is the lower
     bound on the likelihood for the whole corpus.""")


    def std_group(self):
        return self._get_fields('std_')
    def lda_group(self):
        return self._get_fields('lda_')
    def hdp_group(self):
        return self._get_fields('hdp_')
    def ctm_group(self):
        return self._get_fields('ctm_')
    def url_group(self):
        return self._get_fields('url_')
    def upload_group(self):
        return self._get_fields('upload_')
    def toy_group(self):
        return self._get_fields('toy_')
    def arxiv_group(self):
        return self._get_fields('arxiv_')
    def process_group(self):
        return self._get_fields('process_')

    def _get_fields(self, prefix_id):
        filtered_vals = filter(lambda x: x[0].startswith(prefix_id), self.fields.items())
        for field in filtered_vals:
            yield BoundField(self, field[1], field[0])

class PerplexityForm(forms.Form):
    param = forms.CharField(required=False, label = 'parameter',
       widget=Select(
           choices=(
           ('ntopics','Number of Topics'),
           ('someother','')
        )
       ))
    folds = forms.IntegerField(required=True, label='number of folds', initial='1', min_value=1, max_value = 10, widget=forms.TextInput(attrs={"class":"small_int"}), help_text="k-fold perplexity calculation")
    pertest= forms.IntegerField(required=False, label='percent test', initial='20', min_value=1, max_value = 99, widget=forms.TextInput(attrs={"class":"small_int"}), help_text="% of hold-out data if number of folds=1")
    start = forms.IntegerField(required=True, label='parameter range', initial='5', min_value=1,  widget=forms.TextInput(attrs={"class":"small_int"}), help_text="start : increment: end")
    stop = forms.IntegerField(required=True, label='stop', initial='15', widget=forms.TextInput(attrs={"class":"small_int"}))
    step = forms.IntegerField(required=True, label='step', initial='10', min_value=1, widget=forms.TextInput(attrs={"class":"small_int"}))
    current_step = forms.IntegerField(required=False, label="current iteration (ajax helper)")
    current_fold = forms.IntegerField(required=False, label="current fold (ajax helper)")

