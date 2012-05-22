// (C) Copyright 2007, David M. Blei and John D. Lafferty

// This file is part of CTM-C.

// CTM-C is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// CTM-C is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

/*************************************************************************
 *
 * llna.c
 *
 * estimation of an llna model by variational em
 *
 *************************************************************************/

#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <gsl/gsl_blas.h>
#include <assert.h>

#include "corpus.h"
#include "ctm.h"
#include "inference.h"
#include "gsl-wrappers.h"
#include "params.h"

extern llna_params PARAMS;

/*
 * e step
 *
 */

void expectation(corpus* corpus, llna_model* model, llna_ss* ss,
                 double* avg_niter, double* total_lhood,
                 gsl_matrix* corpus_lambda, gsl_matrix* corpus_nu,
                 gsl_matrix* corpus_phi_sum,
                 short reset_var, double* converged_pct)
{
    int i;
    llna_var_param* var;
    doc doc;
    double lhood, total;
    gsl_vector lambda, nu;
    gsl_vector* phi_sum;

    *avg_niter = 0.0;
    *converged_pct = 0;
    phi_sum = gsl_vector_alloc(model->k);
    total = 0;
    for (i = 0; i < corpus->ndocs; i++)
    {   
        if (i % 100 == 0){
        printf("doc %5d\n", i);
    }
        doc = corpus->docs[i];
        var = new_llna_var_param(doc.nterms, model->k);
        if (reset_var)
            init_var_unif(var, &doc, model);
        else
        {
            lambda = gsl_matrix_row(corpus_lambda, i).vector;
            nu= gsl_matrix_row(corpus_nu, i).vector;
            init_var(var, &doc, model, &lambda, &nu);
        }
        lhood = var_inference(var, &doc, model);
        update_expected_ss(var, &doc, ss);
        total += lhood;
        //printf("lhood %5.5e   niter %5d\n", lhood, var->niter);
        *avg_niter += var->niter;
        *converged_pct += var->converged;
        gsl_matrix_set_row(corpus_lambda, i, var->lambda);
        gsl_matrix_set_row(corpus_nu, i, var->nu);
        col_sum(var->phi, phi_sum);
        gsl_matrix_set_row(corpus_phi_sum, i, phi_sum);
        free_llna_var_param(var);
    }
    gsl_vector_free(phi_sum);
    *avg_niter = *avg_niter / corpus->ndocs;
    *converged_pct = *converged_pct / corpus->ndocs;
    *total_lhood = total;
}


/*
 * m step
 *
 */

void cov_shrinkage(gsl_matrix* mle, int n, gsl_matrix* result)
{
    int p = mle->size1, i;
    double temp = 0, alpha = 0, tau = 0, log_lambda_s = 0;
    gsl_vector
        *lambda_star = gsl_vector_calloc(p),
        t, u,
        *eigen_vals = gsl_vector_calloc(p),
        *s_eigen_vals = gsl_vector_calloc(p);
    gsl_matrix
        *d = gsl_matrix_calloc(p,p),
        *eigen_vects = gsl_matrix_calloc(p,p),
        *s_eigen_vects = gsl_matrix_calloc(p,p),
        *result1 = gsl_matrix_calloc(p,p);

    // get eigen decomposition

    sym_eigen(mle, eigen_vals, eigen_vects);
    for (i = 0; i < p; i++)
    {

        // compute shrunken eigenvalues

        temp = 0;
        alpha = 1.0/(n+p+1-2*i);
        vset(lambda_star, i, n * alpha * vget(eigen_vals, i));
    }

    // get diagonal mle and eigen decomposition

    t = gsl_matrix_diagonal(d).vector;
    u = gsl_matrix_diagonal(mle).vector;
    gsl_vector_memcpy(&t, &u);
    sym_eigen(d, s_eigen_vals, s_eigen_vects);

    // compute tau^2

    for (i = 0; i < p; i++)
        log_lambda_s += log(vget(s_eigen_vals, i));
    log_lambda_s = log_lambda_s/p;
    for (i = 0; i < p; i++)
        tau += pow(log(vget(lambda_star, i)) - log_lambda_s, 2)/(p + 4) - 2.0 / n;

    // shrink \lambda* towards the structured eigenvalues

    for (i = 0; i < p; i++)
        vset(lambda_star, i,
             exp((2.0/n)/((2.0/n) + tau) * log_lambda_s +
                 tau/((2.0/n) + tau) * log(vget(lambda_star, i))));

    // put the eigenvalues in a diagonal matrix

    t = gsl_matrix_diagonal(d).vector;
    gsl_vector_memcpy(&t, lambda_star);

    // reconstruct the covariance matrix

    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, d, eigen_vects, 0, result1);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, eigen_vects, result1, 0, result);

    // clean up

    gsl_vector_free(lambda_star);
    gsl_vector_free(eigen_vals);
    gsl_vector_free(s_eigen_vals);
    gsl_matrix_free(d);
    gsl_matrix_free(eigen_vects);
    gsl_matrix_free(s_eigen_vects);
    gsl_matrix_free(result1);
}



void maximization(llna_model* model, llna_ss* ss)
{
    int i, j;
    double sum;

    // mean maximization

    for (i = 0; i < model->k-1; i++)
        vset(model->mu, i, vget(ss->mu_ss, i) / ss->ndata);

    // covariance maximization

    for (i = 0; i < model->k-1; i++)
    {
        for (j = 0; j < model->k-1; j++)
        {
            mset(model->cov, i, j,
                 (1.0 / ss->ndata) *
                 (mget(ss->cov_ss, i, j) +
                  ss->ndata * vget(model->mu, i) * vget(model->mu, j) -
                  vget(ss->mu_ss, i) * vget(model->mu, j) -
                  vget(ss->mu_ss, j) * vget(model->mu, i)));
        }
    }
    if (PARAMS.cov_estimate == SHRINK)
    {
        cov_shrinkage(model->cov, ss->ndata, model->cov);
    }
    matrix_inverse(model->cov, model->inv_cov);
    model->log_det_inv_cov = log_det(model->inv_cov);

    // topic maximization

    for (i = 0; i < model->k; i++)
    {
        sum = 0;
        for (j = 0; j < model->log_beta->size2; j++)
            sum += mget(ss->beta_ss, i, j);

        if (sum == 0) sum = safe_log(sum) * model->log_beta->size2;
        else sum = safe_log(sum);

        for (j = 0; j < model->log_beta->size2; j++)
            mset(model->log_beta, i, j, safe_log(mget(ss->beta_ss, i, j)) - sum);
    }
}


/*
 * run em
 *
 */

llna_model* em_initial_model(int k, corpus* corpus, char* start)
{
    llna_model* model;
    printf("starting from %s\n", start);
    if (strcmp(start, "rand")==0)
        model = random_init(k, corpus->nterms);
    else if (strcmp(start, "seed")==0)
        model = corpus_init(k, corpus);
    else
        model = read_llna_model(start);
    return(model);
}


void em(char* dataset, int k, char* start, char* dir)
{
    FILE* lhood_fptr;
    char string[100];
    int iteration;
    double convergence = 1, lhood = 0, lhood_old = 0;
    corpus* corpus;
    llna_model *model;
    llna_ss* ss;
    time_t t1,t2;
    double avg_niter, converged_pct, old_conv = 0;
    gsl_matrix *corpus_lambda, *corpus_nu, *corpus_phi_sum;
    short reset_var = 1;

    // read the data and make the directory

    corpus = read_data(dataset);
    mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

    // set up the log likelihood log file

    sprintf(string, "%s/likelihood.dat", dir);
    lhood_fptr = fopen(string, "w");

    // run em

    model = em_initial_model(k, corpus, start);
    ss = new_llna_ss(model);
    corpus_lambda = gsl_matrix_alloc(corpus->ndocs, model->k);
    corpus_nu = gsl_matrix_alloc(corpus->ndocs, model->k);
    corpus_phi_sum = gsl_matrix_alloc(corpus->ndocs, model->k);
    time(&t1);
    init_temp_vectors(model->k-1); // !!! hacky
    iteration = 0;
    sprintf(string, "%s/%03d", dir, iteration);
    write_llna_model(model, string);
    do
    {
        printf("***** EM ITERATION %d *****\n", iteration);

        expectation(corpus, model, ss, &avg_niter, &lhood,
                    corpus_lambda, corpus_nu, corpus_phi_sum,
                    reset_var, &converged_pct);
        time(&t2);
        convergence = (lhood_old - lhood) / lhood_old;
        fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld %5.5f %1.5f\n",
                iteration, lhood, convergence, (int) t2 - t1, avg_niter, converged_pct);

        if (((iteration % PARAMS.lag)==0) || isnan(lhood))
        {
            sprintf(string, "%s/%03d", dir, iteration);
            write_llna_model(model, string);
            sprintf(string, "%s/%03d-lambda.dat", dir, iteration);
            printf_matrix(string, corpus_lambda);
            sprintf(string, "%s/%03d-nu.dat", dir, iteration);
            printf_matrix(string, corpus_nu);
        }
        time(&t1);

        if (convergence < 0)
        {
            reset_var = 0;
            if (PARAMS.var_max_iter > 0)
                PARAMS.var_max_iter += 10;
            else PARAMS.var_convergence /= 10;
        }
        else
        {
            maximization(model, ss);
            lhood_old = lhood;
            reset_var = 1;
            iteration++;
        }

        fflush(lhood_fptr);
        reset_llna_ss(ss);
        old_conv = convergence;
    }
    while ((iteration < PARAMS.em_max_iter) &&
           ((convergence > PARAMS.em_convergence) || (convergence < 0)));

    sprintf(string, "%s/final", dir);
    write_llna_model(model, string);
    sprintf(string, "%s/final-lambda.dat", dir);
    printf_matrix(string, corpus_lambda);
    sprintf(string, "%s/final-nu.dat", dir);
    printf_matrix(string, corpus_nu);
    fclose(lhood_fptr);
}


/*
 * load a model, and do approximate inference for each document in a corpus
 *
 */

void inference(char* dataset, char* model_root, char* out)
{
    int i;
    char fname[100];

    // read the data and model
    corpus * corpus = read_data(dataset);
    llna_model * model = read_llna_model(model_root);
    gsl_vector * lhood = gsl_vector_alloc(corpus->ndocs);
    gsl_matrix * corpus_nu = gsl_matrix_alloc(corpus->ndocs, model->k);
    gsl_matrix * corpus_lambda = gsl_matrix_alloc(corpus->ndocs, model->k);
    // gsl_matrix * topic_lhoods = gsl_matrix_alloc(corpus->ndocs, model->k);
    gsl_matrix * phi_sums = gsl_matrix_alloc(corpus->ndocs, model->k);

    // approximate inference
    init_temp_vectors(model->k-1); // !!! hacky
    sprintf(fname, "%s-word-assgn.dat", out);
    FILE* word_assignment_file = fopen(fname, "w");
    for (i = 0; i < corpus->ndocs; i++)
    {
        doc doc = corpus->docs[i];
        llna_var_param * var = new_llna_var_param(doc.nterms, model->k);
        init_var_unif(var, &doc, model);

        vset(lhood, i, var_inference(var, &doc, model));
        gsl_matrix_set_row(corpus_lambda, i, var->lambda);
        gsl_matrix_set_row(corpus_nu, i, var->nu);
        gsl_vector curr_row = gsl_matrix_row(phi_sums, i).vector;
        col_sum(var->phi, &curr_row);
        write_word_assignment(word_assignment_file, &doc, var->phi);

        printf("document %05d, niter = %05d\n", i, var->niter);
        free_llna_var_param(var);
    }

    // output likelihood and some variational parameters
    sprintf(fname, "%s-ctm-lhood.dat", out);
    printf_vector(fname, lhood);
    sprintf(fname, "%s-lambda.dat", out);
    printf_matrix(fname, corpus_lambda);
    sprintf(fname, "%s-nu.dat", out);
    printf_matrix(fname, corpus_nu);
    sprintf(fname, "%s-phi-sum.dat", out);
    printf_matrix(fname, phi_sums);

}


/*
 * split documents into two random parts
 *
 */

void within_doc_split(char* dataset, char* src_data, char* dest_data, double prop)
{
    int i;
    corpus * corp, * dest_corp;

    corp = read_data(dataset);
    dest_corp = malloc(sizeof(corpus));
    printf("splitting %d docs\n", corp->ndocs);
    dest_corp->docs = malloc(sizeof(doc) * corp->ndocs);
    dest_corp->nterms = corp->nterms;
    dest_corp->ndocs = corp->ndocs;
    for (i = 0; i < corp->ndocs; i++)
        split(&(corp->docs[i]), &(dest_corp->docs[i]), prop);
    write_corpus(dest_corp, dest_data);
    write_corpus(corp, src_data);
}


/*
 * for each partially observed document: (a) perform inference on the
 * observations (b) take expected theta and compute likelihood
 *
 */

int pod_experiment(char* observed_data, char* heldout_data,
                   char* model_root, char* out)
{
    corpus *obs, *heldout;
    llna_model *model;
    llna_var_param *var;
    int i;
    gsl_vector *log_lhood, *e_theta;
    doc obs_doc, heldout_doc;
    char string[100];
    double total_lhood = 0, total_words = 0, l;
    FILE* e_theta_file = fopen("/Users/blei/llna050_e_theta.txt", "w");

    // load model and data
    obs = read_data(observed_data);
    heldout = read_data(heldout_data);
    assert(obs->ndocs == heldout->ndocs);
    model = read_llna_model(model_root);

    // run experiment
    init_temp_vectors(model->k-1); // !!! hacky
    log_lhood = gsl_vector_alloc(obs->ndocs + 1);
    e_theta = gsl_vector_alloc(model->k);
    for (i = 0; i < obs->ndocs; i++)
    {
        // get observed and heldout documents
        obs_doc = obs->docs[i];
        heldout_doc = heldout->docs[i];
        // compute variational distribution
        var = new_llna_var_param(obs_doc.nterms, model->k);
        init_var_unif(var, &obs_doc, model);
        var_inference(var, &obs_doc, model);
        expected_theta(var, &obs_doc, model, e_theta);

        vfprint(e_theta, e_theta_file);

        // approximate inference of held out data
        l = log_mult_prob(&heldout_doc, e_theta, model->log_beta);
        vset(log_lhood, i, l);
        total_words += heldout_doc.total;
        total_lhood += l;
        printf("hid doc %d    log_lhood %5.5f\n", i, vget(log_lhood, i));
        // save results?
        free_llna_var_param(var);
    }
    vset(log_lhood, obs->ndocs, exp(-total_lhood/total_words));
    printf("perplexity : %5.10f", exp(-total_lhood/total_words));
    sprintf(string, "%s-pod-llna.dat", out);
    printf_vector(string, log_lhood);
    return(0);
}


/*
 * little function to count the words in each document and spit it out
 *
 */

void count(char* corpus_name, char* output_name)
{
    corpus *c;
    int i;
    FILE *f;
    int j;
    f = fopen(output_name, "w");
    c = read_data(corpus_name);
    for (i = 0; i < c->ndocs; i++)
    {
        j = c->docs[i].total;
        fprintf(f, "%5d\n", j);
    }
}

/*
 * main function
 *
 */

int main(int argc, char* argv[])
{
    if (argc > 1)
    {
        if (strcmp(argv[1], "est")==0)
        {
            read_params(argv[6]);
            print_params();
            em(argv[2], atoi(argv[3]), argv[4], argv[5]);
            return(0);
        }
        if (strcmp(argv[1], "inf")==0)
        {
            read_params(argv[5]);
            print_params();
            inference(argv[2], argv[3], argv[4]);
            return(0);
        }
    }
    printf("usage : ctm est <dataset> <# topics> <rand/seed/model> <dir> <settings>\n");
    printf("        ctm inf <dataset> <model-prefix> <results-prefix> <settings>\n");
    return(0);
}
