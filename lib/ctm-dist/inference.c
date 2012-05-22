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

#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <assert.h>

#include "gsl-wrappers.h"
#include "corpus.h"
#include "ctm.h"
#include "params.h"
#include "inference.h"

extern llna_params PARAMS;

double f_lambda(const gsl_vector * p, void * params);
void df_lambda(const gsl_vector * p, void * params, gsl_vector * df);
void fdf_lambda(const gsl_vector * p, void * params, double * f, gsl_vector * df);

double f_nu(const gsl_vector * p, void * params);
void df_nu(const gsl_vector * p, void * params, gsl_vector * df);
void fdf_nu(const gsl_vector * p, void * params, double * f, gsl_vector * df);

/*
 * temporary k-1 vectors so we don't have to allocate, deallocate
 *
 */

gsl_vector ** temp;
int ntemp = 4;

void init_temp_vectors(int size)
{
    int i;
    temp = malloc(sizeof(gsl_vector *)*ntemp);
    for (i = 0; i < 4; i++)
        temp[i] = gsl_vector_alloc(size);
}


/*
 * likelihood bound
 *
 */

double expect_mult_norm(llna_var_param * var)
{
    int i;
    double sum_exp = 0;
    int niter = var->lambda->size;

    for (i = 0; i < niter; i++)
        sum_exp += exp(vget(var->lambda, i) + (0.5) * vget(var->nu,i));

    return((1.0/var->zeta) * sum_exp - 1.0 + log(var->zeta));
}


void lhood_bnd(llna_var_param* var, doc* doc, llna_model* mod)
{
    int i = 0, j = 0, k = mod->k;
    gsl_vector_set_zero(var->topic_scores);

    // E[log p(\eta | \mu, \Sigma)] + H(q(\eta | \lambda, \nu)

    double lhood  = (0.5) * mod->log_det_inv_cov + (0.5) * (mod->k-1);
    for (i = 0; i < k-1; i++)
    {
        double v = - (0.5) * vget(var->nu, i) * mget(mod->inv_cov,i, i);
        for (j = 0; j < mod->k-1; j++)
        {
            v -= (0.5) *
                (vget(var->lambda, i) - vget(mod->mu, i)) *
                mget(mod->inv_cov, i, j) *
                (vget(var->lambda, j) - vget(mod->mu, j));
        }
        v += (0.5) * log(vget(var->nu, i));
        lhood += v;
    }

    // E[log p(z_n | \eta)] + E[log p(w_n | \beta)] + H(q(z_n | \phi_n))

    lhood -= expect_mult_norm(var) * doc->total;
    for (i = 0; i < doc->nterms; i++)
    {
        // !!! we can speed this up by turning it into a dot product
        // !!! profiler says this is where some time is spent
        for (j = 0; j < mod->k; j++)
        {
            double phi_ij = mget(var->phi, i, j);
            double log_phi_ij = mget(var->log_phi, i, j);
            if (phi_ij > 0)
            {
                vinc(var->topic_scores, j, phi_ij * doc->count[i]);
                lhood +=
                    doc->count[i] * phi_ij *
                    (vget(var->lambda, j) +
                     mget(mod->log_beta, j, doc->word[i]) -
                     log_phi_ij);
            }
        }
    }
    var->lhood = lhood;
    assert(!isnan(var->lhood));
}


/**
 * optimize zeta
 *
 */

int opt_zeta(llna_var_param * var, doc * doc, llna_model * mod)
{
    int i;

    var->zeta = 1.0;
    for (i = 0; i < mod->k-1; i++)
        var->zeta += exp(vget(var->lambda, i) + (0.5) * vget(var->nu, i));

    return(0);
}


/**
 * optimize phi
 *
 */

void opt_phi(llna_var_param * var, doc * doc, llna_model * mod)
{
    int i, n, K = mod->k;
    double log_sum_n = 0;

    // compute phi proportions in log space

    for (n = 0; n < doc->nterms; n++)
    {
        log_sum_n = 0;
        for (i = 0; i < K; i++)
        {
            mset(var->log_phi, n, i,
                 vget(var->lambda, i) + mget(mod->log_beta, i, doc->word[n]));
            if (i == 0)
                log_sum_n = mget(var->log_phi, n, i);
            else
                log_sum_n =  log_sum(log_sum_n, mget(var->log_phi, n, i));
        }
        for (i = 0; i < K; i++)
        {
            mset(var->log_phi, n, i, mget(var->log_phi, n, i) - log_sum_n);
            mset(var->phi, n, i, exp(mget(var->log_phi, n, i)));
        }
    }
}

/**
 * optimize lambda
 *
 */

void fdf_lambda(const gsl_vector * p, void * params, double * f, gsl_vector * df)
{
    *f = f_lambda(p, params);
    df_lambda(p, params, df);
}


double f_lambda(const gsl_vector * p, void * params)
{
    double term1, term2, term3;
    int i;
    llna_var_param * var = ((bundle *) params)->var;
    doc * doc = ((bundle *) params)->doc;
    llna_model * mod = ((bundle *) params)->mod;

    // compute lambda^T \sum phi
    gsl_blas_ddot(p,((bundle *) params)->sum_phi, &term1);
    // compute lambda - mu (= temp1)
    gsl_blas_dcopy(p, temp[1]);
    gsl_blas_daxpy (-1.0, mod->mu, temp[1]);
    // compute (lambda - mu)^T Sigma^-1 (lambda - mu)
    gsl_blas_dsymv(CblasUpper, 1, mod->inv_cov, temp[1], 0, temp[2]);
    // gsl_blas_dgemv(CblasNoTrans, 1, mod->inv_cov, temp[1], 0, temp[2]);
    gsl_blas_ddot(temp[2], temp[1], &term2);
    term2 = - 0.5 * term2;
    // last term
    term3 = 0;
    for (i = 0; i < mod->k-1; i++)
        term3 += exp(vget(p, i) + (0.5) * vget(var->nu,i));
    term3 = -((1.0/var->zeta) * term3 - 1.0 + log(var->zeta)) * doc->total;
    // negate for minimization
    return(-(term1+term2+term3));
}


void df_lambda(const gsl_vector * p, void * params, gsl_vector * df)
{
    // cast bundle {variational parameters, model, document}

    llna_var_param * var = ((bundle *) params)->var;
    doc * doc = ((bundle *) params)->doc;
    llna_model * mod = ((bundle *) params)->mod;
    gsl_vector * sum_phi = ((bundle *) params)->sum_phi;

    // compute \Sigma^{-1} (\mu - \lambda)

    gsl_vector_set_zero(temp[0]);
    gsl_blas_dcopy(mod->mu, temp[1]);
    gsl_vector_sub(temp[1], p);
    gsl_blas_dsymv(CblasLower, 1, mod->inv_cov, temp[1], 0, temp[0]);

    // compute - (N / \zeta) * exp(\lambda + \nu^2 / 2)

    int i;
    for (i = 0; i < temp[3]->size; i++)
    {
        vset(temp[3], i, -(((double) doc->total) / var->zeta) *
             exp(vget(p, i) + 0.5 * vget(var->nu, i)));
    }

    // set return value (note negating derivative of bound)

    gsl_vector_set_all(df, 0.0);
    gsl_vector_sub(df, temp[0]);
    gsl_vector_sub(df, sum_phi);
    gsl_vector_sub(df, temp[3]);
}


int opt_lambda(llna_var_param * var, doc * doc, llna_model * mod)
{
    gsl_multimin_function_fdf lambda_obj;
    const gsl_multimin_fdfminimizer_type * T;
    gsl_multimin_fdfminimizer * s;
    bundle b;
    int iter = 0, i, j;
    int status;
    double f_old, converged;

    b.var = var;
    b.doc = doc;
    b.mod = mod;

    // precompute \sum_n \phi_n and put it in the bundle

    b.sum_phi = gsl_vector_alloc(mod->k-1);
    gsl_vector_set_zero(b.sum_phi);
    for (i = 0; i < doc->nterms; i++)
    {
        for (j = 0; j < mod->k-1; j++)
        {
            vset(b.sum_phi, j,
                 vget(b.sum_phi, j) +
                 ((double) doc->count[i]) * mget(var->phi, i, j));
        }
    }

    lambda_obj.f = &f_lambda;
    lambda_obj.df = &df_lambda;
    lambda_obj.fdf = &fdf_lambda;
    lambda_obj.n = mod->k-1;
    lambda_obj.params = (void *)&b;

    // starting value
    // T = gsl_multimin_fdfminimizer_vector_bfgs;
    T = gsl_multimin_fdfminimizer_conjugate_fr;
    // T = gsl_multimin_fdfminimizer_steepest_descent;
    s = gsl_multimin_fdfminimizer_alloc (T, mod->k-1);

    gsl_vector* x = gsl_vector_calloc(mod->k-1);
    for (i = 0; i < mod->k-1; i++) vset(x, i, vget(var->lambda, i));
    gsl_multimin_fdfminimizer_set (s, &lambda_obj, x, 0.01, 1e-3);
    do
    {
        iter++;
        f_old = s->f;
        status = gsl_multimin_fdfminimizer_iterate (s);
        converged = fabs((f_old - s->f) / f_old);
        // printf("f(lambda) = %5.17e ; conv = %5.17e\n", s->f, converged);
        if (status) break;
        status = gsl_multimin_test_gradient (s->gradient, PARAMS.cg_convergence);
    }
    while ((status == GSL_CONTINUE) &&
           ((PARAMS.cg_max_iter < 0) || (iter < PARAMS.cg_max_iter)));
    // while ((converged > PARAMS.cg_convergence) &&
    // ((PARAMS.cg_max_iter < 0) || (iter < PARAMS.cg_max_iter)));
    if (iter == PARAMS.cg_max_iter)
        printf("warning: cg didn't converge (lambda) \n");

    for (i = 0; i < mod->k-1; i++)
        vset(var->lambda, i, vget(s->x, i));
    vset(var->lambda, i, 0);

    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(b.sum_phi);
    gsl_vector_free(x);

    return(0);
}

/**
 * optimize nu
 *
 */

double f_nu_i(double nu_i, int i, llna_var_param * var,
              llna_model * mod, doc * d)
{
    double v;

    v = - (nu_i * mget(mod->inv_cov, i, i) * 0.5)
        - (((double) d->total/var->zeta) * exp(vget(var->lambda, i) + nu_i/2))
        + (0.5 * safe_log(nu_i));

    return(v);
}


double df_nu_i(double nu_i, int i, llna_var_param * var,
               llna_model * mod, doc * d)
{
    double v;

    v = - (mget(mod->inv_cov, i, i) * 0.5)
        - (0.5 * ((double) d->total/var->zeta) * exp(vget(var->lambda, i) + nu_i/2))
        + (0.5 * (1.0 / nu_i));

    return(v);
}


double d2f_nu_i(double nu_i, int i, llna_var_param * var, llna_model * mod, doc * d)
{
    double v;

    v = - (0.25 * ((double) d->total/var->zeta) * exp(vget(var->lambda, i) + nu_i/2))
        - (0.5 * (1.0 / (nu_i * nu_i)));

    return(v);
}


void opt_nu(llna_var_param * var, doc * d, llna_model * mod)
{
    int i;

    // !!! here i changed to k-1
    for (i = 0; i < mod->k-1; i++)
        opt_nu_i(i, var, mod, d);
}


double fixed_point_iter_i(int i, llna_var_param * var, llna_model * mod, doc * d)
{
    double v;
    double lambda = vget(var->lambda, i);
    double nu = vget(var->nu, i);
    double c = ((double) d->total / var->zeta);

    v = mget(mod->inv_cov,i,i) + c * exp(lambda + nu/2);

    return(v);
}


void opt_nu_i(int i, llna_var_param * var, llna_model * mod, doc * d)
{
    double init_nu = 10;
    double nu_i = 0, log_nu_i = 0, df = 0, d2f = 0;
    int iter = 0;

    log_nu_i = log(init_nu);
    do
    {
        iter++;
        nu_i = exp(log_nu_i);
        // assert(!isnan(nu_i));
        if (isnan(nu_i))
        {
            init_nu = init_nu*2;
            printf("warning : nu is nan; new init = %5.5f\n", init_nu);
            log_nu_i = log(init_nu);
            nu_i = init_nu;
        }
        // f = f_nu_i(nu_i, i, var, mod, d);
        // printf("%5.5f  %5.5f \n", nu_i, f);
        df = df_nu_i(nu_i, i, var, mod, d);
        d2f = d2f_nu_i(nu_i, i, var, mod, d);
        log_nu_i = log_nu_i - (df*nu_i)/(d2f*nu_i*nu_i+df*nu_i);
    }
    while (fabs(df) > NEWTON_THRESH);

    vset(var->nu, i, exp(log_nu_i));
}

/**
 * initial variational parameters
 *
 */

void init_var_unif(llna_var_param * var, doc * doc, llna_model * mod)
{
    int i;

    gsl_matrix_set_all(var->phi, 1.0/mod->k);
    gsl_matrix_set_all(var->log_phi, -log((double) mod->k));
    var->zeta = 10;
    for (i = 0; i < mod->k-1; i++)
    {
        vset(var->nu, i, 10.0);
        vset(var->lambda, i, 0);
    }
    vset(var->nu, i, 0);
    vset(var->lambda, i, 0);
    var->niter = 0;
    var->lhood = 0;
}


void init_var(llna_var_param * var, doc * doc, llna_model * mod, gsl_vector *lambda, gsl_vector *nu)
{
    gsl_vector_memcpy(var->lambda, lambda);
    gsl_vector_memcpy(var->nu, nu);
    opt_zeta(var, doc, mod);
    opt_phi(var, doc, mod);
    var->niter = 0;
}




/**
 *
 * variational inference
 *
 */

llna_var_param * new_llna_var_param(int nterms, int k)
{
    llna_var_param * ret = malloc(sizeof(llna_var_param));
    ret->lambda = gsl_vector_alloc(k);
    ret->nu = gsl_vector_alloc(k);
    ret->phi = gsl_matrix_alloc(nterms, k);
    ret->log_phi = gsl_matrix_alloc(nterms, k);
    ret->zeta = 0;
    ret->topic_scores = gsl_vector_alloc(k);
    return(ret);
}


void free_llna_var_param(llna_var_param * v)
{
    gsl_vector_free(v->lambda);
    gsl_vector_free(v->nu);
    gsl_matrix_free(v->phi);
    gsl_matrix_free(v->log_phi);
    gsl_vector_free(v->topic_scores);
    free(v);
}


double var_inference(llna_var_param * var,
                     doc * doc,
                     llna_model * mod)
{
    double lhood_old = 0;
    double convergence;

    lhood_bnd(var, doc, mod);
    do
    {
        var->niter++;

        opt_zeta(var, doc, mod);
        opt_lambda(var, doc, mod);
        opt_zeta(var, doc, mod);
        opt_nu(var, doc, mod);
        opt_zeta(var, doc, mod);
        opt_phi(var, doc, mod);

        lhood_old = var->lhood;
        lhood_bnd(var, doc, mod);

        convergence = fabs((lhood_old - var->lhood) / lhood_old);
        // printf("lhood = %8.6f (%7.6f)\n", var->lhood, convergence);

        if ((lhood_old > var->lhood) && (var->niter > 1))
            printf("WARNING: iter %05d %5.5f > %5.5f\n",
                   var->niter, lhood_old, var->lhood);
    }
    while ((convergence > PARAMS.var_convergence) &&
           ((PARAMS.var_max_iter < 0) || (var->niter < PARAMS.var_max_iter)));

    if (convergence > PARAMS.var_convergence) var->converged = 0;
    else var->converged = 1;

    return(var->lhood);
}


void update_expected_ss(llna_var_param* var, doc* d, llna_ss* ss)
{
    int i, j, w, c;
    double lilj;

    // covariance and mean suff stats
    for (i = 0; i < ss->cov_ss->size1; i++)
    {
        vinc(ss->mu_ss, i, vget(var->lambda, i));
        for (j = 0; j < ss->cov_ss->size2; j++)
        {
            lilj = vget(var->lambda, i) * vget(var->lambda, j);
            if (i==j)
                mset(ss->cov_ss, i, j,
                     mget(ss->cov_ss, i, j) + vget(var->nu, i) + lilj);
            else
                mset(ss->cov_ss, i, j, mget(ss->cov_ss, i, j) + lilj);
        }
    }
    // topics suff stats
    for (i = 0; i < d->nterms; i++)
    {
        for (j = 0; j < ss->beta_ss->size1; j++)
        {
            w = d->word[i];
            c = d->count[i];
            mset(ss->beta_ss, j, w,
                 mget(ss->beta_ss, j, w) + c * mget(var->phi, i, j));
        }
    }
    // number of data
    ss->ndata++;
}

/*
 * importance sampling the likelihood based on the variational posterior
 *
 */

double sample_term(llna_var_param* var, doc* d, llna_model* mod, double* eta)
{
    int i, j, n;
    double t1, t2, sum, theta[mod->k];
    double word_term;

    t1 = (0.5) * mod->log_det_inv_cov;
    t1 += - (0.5) * (mod->k) * 1.837877;
    for (i = 0; i < mod->k; i++)
        for (j = 0; j < mod->k ; j++)
            t1 -= (0.5) *
                (eta[i] - vget(mod->mu, i)) *
                mget(mod->inv_cov, i, j) *
                (eta[j] - vget(mod->mu, j));

    // compute theta
    sum = 0;
    for (i = 0; i < mod->k; i++)
    {
        theta[i] = exp(eta[i]);
        sum += theta[i];
    }
    for (i = 0; i < mod->k; i++)
    {
        theta[i] = theta[i] / sum;
    }

    // compute word probabilities
    for (n = 0; n < d->nterms; n++)
    {
        word_term = 0;
        for (i = 0; i < mod->k; i++)
            word_term += theta[i]*exp(mget(mod->log_beta,i,d->word[n]));
        t1 += d->count[n] * safe_log(word_term);
    }

    // log(q(\eta | lambda, nu))
    t2 = 0;
    for (i = 0; i < mod->k; i++)
        t2 += log(gsl_ran_gaussian_pdf(eta[i] - vget(var->lambda,i), sqrt(vget(var->nu,i))));
    return(t1-t2);
}


double sample_lhood(llna_var_param* var, doc* d, llna_model* mod)
{
    int nsamples, i, n;
    double eta[mod->k];
    double log_prob, sum = 0, v;
    gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);

    gsl_rng_set(r, (long) 1115574245);
    nsamples = 10000;

    // for each sample
    for (n = 0; n < nsamples; n++)
    {
        // sample eta from q(\eta)
        for (i = 0; i < mod->k; i++)
        {
            v = gsl_ran_gaussian_ratio_method(r, sqrt(vget(var->nu,i)));
            eta[i] = v + vget(var->lambda, i);
        }
        // compute p(w | \eta) - q(\eta)
        log_prob = sample_term(var, d, mod, eta);
        // update log sum
        if (n == 0) sum = log_prob;
        else sum = log_sum(sum, log_prob);
        // printf("%5.5f\n", (sum - log(n+1)));
    }
    sum = sum - log((double) nsamples);
    return(sum);
}


/*
 * expected theta under a variational distribution
 *
 * (v is assumed allocated to the right length.)
 *
 */


void expected_theta(llna_var_param *var, doc* d, llna_model *mod, gsl_vector* val)
{
    int nsamples, i, n;
    double eta[mod->k];
    double theta[mod->k];
    double e_theta[mod->k];
    double sum, w, v;
    gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);

    gsl_rng_set(r, (long) 1115574245);
    nsamples = 100;

    // initialize e_theta
    for (i = 0; i < mod->k; i++) e_theta[i] = -1;
    // for each sample
    for (n = 0; n < nsamples; n++)
    {
        // sample eta from q(\eta)
        for (i = 0; i < mod->k; i++)
        {
            v = gsl_ran_gaussian_ratio_method(r, sqrt(vget(var->nu,i)));
            eta[i] = v + vget(var->lambda, i);
        }
        // compute p(w | \eta) - q(\eta)
        w = sample_term(var, d, mod, eta);
        // compute theta
        sum = 0;
        for (i = 0; i < mod->k; i++)
        {
            theta[i] = exp(eta[i]);
            sum += theta[i];
        }
        for (i = 0; i < mod->k; i++)
            theta[i] = theta[i] / sum;
        // update e_theta
        for (i = 0; i < mod->k; i++)
            e_theta[i] = log_sum(e_theta[i], w +  safe_log(theta[i]));
    }
    // normalize e_theta and set return vector
    sum = -1;
    for (i = 0; i < mod->k; i++)
    {
        e_theta[i] = e_theta[i] - log(nsamples);
        sum = log_sum(sum, e_theta[i]);
    }
    for (i = 0; i < mod->k; i++)
        vset(val, i, exp(e_theta[i] - sum));
}

/*
 * log probability of the document under proportions theta and topics
 * beta
 *
 */

double log_mult_prob(doc* d, gsl_vector* theta, gsl_matrix* log_beta)
{
    int i, k;
    double ret = 0;
    double term_prob;

    for (i = 0; i < d->nterms; i++)
    {
        term_prob = 0;
        for (k = 0; k < log_beta->size1; k++)
            term_prob += vget(theta, k) * exp(mget(log_beta, k, d->word[i]));
        ret = ret + safe_log(term_prob) * d->count[i];
    }
    return(ret);
}


/*
 * writes the word assignments line for a document to a file
 *
 */

void write_word_assignment(FILE* f, doc* d, gsl_matrix* phi)
{
    int n;

    fprintf(f, "%03d", d->nterms);
    for (n = 0; n < d->nterms; n++)
    {
        gsl_vector phi_row = gsl_matrix_row(phi, n).vector;
        fprintf(f, " %04d:%02d", d->word[n], argmax(&phi_row));
    }
    fprintf(f, "\n");
    fflush(f);
}
