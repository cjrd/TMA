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

#ifndef LLNA_INFERENCE_H
#define LLNA_INFERENCE_H

#define NEWTON_THRESH 1e-10

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <stdlib.h>
#include <stdio.h>

#include "corpus.h"
#include "ctm.h"
#include "gsl-wrappers.h"

typedef struct llna_var_param {
    gsl_vector * nu;
    gsl_vector * lambda;
    double zeta;
    gsl_matrix * phi;
    gsl_matrix * log_phi;
    int niter;
    short converged;
    double lhood;
    gsl_vector * topic_scores;
} llna_var_param;


typedef struct bundle {
    llna_var_param * var;
    llna_model * mod;
    doc * doc;
    gsl_vector * sum_phi;
} bundle;


/*
 * functions
 *
 */

void init_temp_vectors(int size);
int opt_lambda(llna_var_param * var, doc * doc, llna_model * mod);
void opt_phi(llna_var_param * var, doc * doc, llna_model * mod);
void opt_nu(llna_var_param * var, doc * doc, llna_model * mod);
int opt_zeta(llna_var_param * var, doc * doc, llna_model * mod);
void lhood_bnd(llna_var_param *var, doc* doc, llna_model* mod);
double var_inference(llna_var_param * var, doc * doc, llna_model * mod);
llna_var_param* new_llna_var_param(int, int);
void free_llna_var_param(llna_var_param *);
void update_expected_ss(llna_var_param* , doc*, llna_ss*);
void init_var_unif(llna_var_param * var, doc * doc, llna_model * mod);
void init_var(llna_var_param *var, doc *doc, llna_model *mod, gsl_vector *lambda, gsl_vector *nu);
void opt_nu_i(int i, llna_var_param * var, llna_model * mod, doc * d);
double fixed_point_iter_i(int, llna_var_param *, llna_model *, doc *);
double sample_lhood(llna_var_param* var, doc* d, llna_model* mod);
void expected_theta(llna_var_param *var, doc* d, llna_model *mod, gsl_vector* v);
double log_mult_prob(doc* d, gsl_vector* theta, gsl_matrix* log_beta);
void write_word_assignment(FILE* f, doc* d, gsl_matrix* phi);

#endif
