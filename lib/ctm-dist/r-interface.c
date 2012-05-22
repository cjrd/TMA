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

// NOTE: this file is not currently used

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include "gsl-wrappers.h"
#include "ctm.h"
#include "inference.h"
#include "params.h"

llna_model global_mod;
extern llna_params PARAMS;

/*
 * for translating r arrays into gsl matrices and vectors
 *
 */

double r_mtx_get(const double* v, int i, int j, int nrow)
{
    return(v[nrow * j + i]);
}


gsl_matrix * r_to_gsl_matrix(const double * v, int nrow, int ncol)
{
    int i, j;
    gsl_matrix * ret;

    ret = gsl_matrix_alloc(nrow, ncol);
    for (i = 0; i < nrow; i++)
    {
	for (j = 0; j < ncol; j++)
	{
	    mset(ret, i, j, r_mtx_get(v, i, j, nrow));
	}
    }
    return(ret);
}


gsl_vector * r_to_gsl_vector(const double * v, int size)
{
    int i;
    gsl_vector * ret;

    ret = gsl_vector_alloc(size);
    for (i = 0; i < size; i++)
    {
	vset(ret, i, v[i]);
    }
    return(ret);
}


/*
 * sets the global model
 *
 */

void r_set_mod(int * k, int * nterms,
               double * mu,
               double * inv_cov,
               double * cov,
               double * log_det_inv_cov,
               double * log_beta)
{
    global_mod.k = *k;
    global_mod.log_beta = r_to_gsl_matrix(log_beta, global_mod.k, *nterms);
    global_mod.mu = r_to_gsl_vector(mu, global_mod.k-1);
    global_mod.inv_cov = r_to_gsl_matrix(inv_cov, global_mod.k-1, global_mod.k-1);
    global_mod.cov = r_to_gsl_matrix(cov, global_mod.k-1, global_mod.k-1);
    global_mod.log_det_inv_cov = *log_det_inv_cov;
    init_temp_vectors(global_mod.k-1);
    default_params();
}


/*
 * compute the likelihood bound for variational parameters and document
 *
 */

void r_lhood_bound(double * lambda, double * nu, double * phi, double * zeta,
                   int * word, int * count, int * total, int * nterms,
                   double * val)
{
    llna_var_param var;
    doc doc;
    int i;

    var.lambda = r_to_gsl_vector(lambda, global_mod.k-1);
    var.nu = r_to_gsl_vector(nu, global_mod.k-1);
    var.phi = r_to_gsl_matrix(phi, *nterms, global_mod.k);
    var.zeta = *zeta;

    doc.nterms = *nterms;
    doc.total = *total;
    doc.count = count;
    doc.word = word;
    for (i = 0; i < doc.nterms; i++)
	doc.word[i] = doc.word[i] - 1;

    lhood_bnd(&var, &doc, &global_mod);
}


/*
 * variational inference given a document and pointers to variational
 * parameters
 *
 */

void r_var_inference(double * lambda, double * nu, double * phi,
                     double * zeta,
                     int * word, int * count, int * total, int * nterms,
                     double * lhood)
{
    llna_var_param * var;
    doc doc;
    int i, j, n;
    // set up document
    doc.nterms = *nterms;
    doc.total = *total;
    doc.count = count;
    doc.word = word;
    // !!! note we assume that the words are 1-indexed coming from R
    for (i = 0; i < doc.nterms; i++)
	doc.word[i] = doc.word[i] - 1;
    // allocate variational parameters
    var = new_llna_var_param(*nterms, global_mod.k);
    // run variational inference
    lhood[0] = var_inference(var, &doc, &global_mod);
    init_var_unif(var, &doc, &global_mod);
    printf("LHOOD BOUND : %5.5f\n", lhood[0]);
    // return variational parameters
    *zeta = var->zeta;
    for (i = 0; i < global_mod.k-1; i++)
    {
	lambda[i] = vget(var->lambda, i);
	nu[i] = vget(var->nu, i);
    }
    j = 0;
    for (i = 0; i < global_mod.k; i++)
    {
	for (n = 0; n < doc.nterms; n++)
	{
	    phi[j] = mget(var->phi, n, i);
	    j++;
	}
    }
    // clean up
    free_llna_var_param(var);
}
