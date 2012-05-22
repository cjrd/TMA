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

#include "gsl-wrappers.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>

double safe_log(double x)
{
    if (x == 0) return(-1000);
    else return(log(x));
}


double log_sum(double log_a, double log_b)
{
  double v;

  if (log_a == -1) return(log_b);

  if (log_a < log_b)
  {
      v = log_b+log(1 + exp(log_a-log_b));
  }
  else
  {
      v = log_a+log(1 + exp(log_b-log_a));
  }
  return(v);
}


double vget(const gsl_vector* v, int i)
{
    return(gsl_vector_get(v, i));
}


void vset(gsl_vector* v, int i, double x)
{
    gsl_vector_set(v, i, x);
}


void vinc(gsl_vector* v, int i, double x)
{
    vset(v, i, vget(v, i) + x);
}


double mget(const gsl_matrix* m, int i, int j)
{
    return(gsl_matrix_get(m, i, j));
}


void mset(gsl_matrix* m, int i, int j, double x)
{
    gsl_matrix_set(m, i, j, x);
}


void minc(gsl_matrix* m, int i, int j, double x)
{
    mset(m, i, j, mget(m, i, j) + x);
}


void col_sum(gsl_matrix* m, gsl_vector* val)
{
    int i, j;
    gsl_vector_set_all(val, 0);

    for (i = 0; i < m->size1; i++)
        for (j = 0; j < m->size2; j++)
            vinc(val, j, mget(m, i, j));
}


void vprint(const gsl_vector * v)
{
    int i;
    for (i = 0; i < v->size; i++)
	printf("%5.5f ", vget(v, i));
    printf("\n\n");
}


void vfprint(const gsl_vector * v, FILE * f)
{
    int i;
    for (i = 0; i < v->size; i++)
	fprintf(f, "%5.5f ", vget(v, i));
    fprintf(f, "\n");
    fflush(f);
}



void mprint(const gsl_matrix * m)
{
    int i, j;
    for (i = 0; i < m->size1; i++)
    {
	for (j = 0; j < m->size2; j++)
	    printf("%5.5f ", mget(m, i, j));
	printf("\n");
    }
}


void scanf_vector(char* filename, gsl_vector* v)
{
    FILE* fileptr;
    fileptr = fopen(filename, "r");
    gsl_vector_fscanf(fileptr, v);
    fclose(fileptr);
}


void scanf_matrix(char* filename, gsl_matrix * m)
{
    FILE* fileptr;
    fileptr = fopen(filename, "r");
    gsl_matrix_fscanf(fileptr, m);
    fclose(fileptr);
}

void printf_vector(char* filename, gsl_vector* v)
{
    FILE* fileptr;
    fileptr = fopen(filename, "w");
    gsl_vector_fprintf(fileptr, v, "%f");
    fclose(fileptr);
}


void printf_matrix(char* filename, gsl_matrix * m)
{
    FILE* fileptr;
    fileptr = fopen(filename, "w");
    gsl_matrix_fprintf(fileptr, m, "%f");
    fclose(fileptr);
}


void matrix_inverse(gsl_matrix* m, gsl_matrix* inverse)
{
    gsl_matrix *lu;
    gsl_permutation* p;
    int signum;

    p = gsl_permutation_alloc(m->size1);
    lu = gsl_matrix_alloc(m->size1, m->size2);

    gsl_matrix_memcpy(lu, m);
    gsl_linalg_LU_decomp(lu, p, &signum);
    gsl_linalg_LU_invert(lu, p, inverse);

    gsl_matrix_free(lu);
    gsl_permutation_free(p);
}


double log_det(gsl_matrix* m)
{
    gsl_matrix* lu;
    gsl_permutation* p;
    double result;
    int signum;

    p = gsl_permutation_alloc(m->size1);
    lu = gsl_matrix_alloc(m->size1, m->size2);

    gsl_matrix_memcpy(lu, m);
    gsl_linalg_LU_decomp(lu, p, &signum);
    result = gsl_linalg_LU_lndet(lu);

    gsl_matrix_free(lu);
    gsl_permutation_free(p);

    return(result);
}


void sym_eigen(gsl_matrix* m, gsl_vector* vals, gsl_matrix* vects)
{
    gsl_eigen_symmv_workspace* wk;
    gsl_matrix* mcpy;
    int r;

    mcpy = gsl_matrix_alloc(m->size1, m->size2);
    wk = gsl_eigen_symmv_alloc(m->size1);
    gsl_matrix_memcpy(mcpy, m);
    r = gsl_eigen_symmv(mcpy, vals, vects, wk);
    gsl_eigen_symmv_free(wk);
    gsl_matrix_free(mcpy);
}


double sum(gsl_vector* v)
{
    double *data = v->data, val = 0;
    int size = v->size;
    int i;
    for (i = 0; i < size; i++)
        val += data[i];
    return(val);
}


void center(gsl_vector* v)
{
    int size = v->size;
    double mean = sum(v)/size;
    int i;
    for (i = 0; i < size; i++)
        vset(v, i, vget(v,i)-mean);
}


void normalize(gsl_vector* v)
{
    int size = v->size;
    double sum_v = sum(v);
    int i;
    for (i = 0; i < size; i++)
        vset(v, i, vget(v,i)/sum_v);
}


double norm(gsl_vector *v)
{
    double val = 0;
    int i;

    for (i = 0; i < v->size; i++)
        val += vget(v, i) * vget(v, i);
    return(sqrt(val));
}


int argmax(gsl_vector *v)
{
    int argmax = 0;
    double max = vget(v, 0);
    int i;
    for (i = 1; i < v->size; i++)
    {
        double val = vget(v, i);
        if (val > max)
        {
            argmax = i;
            max    = val;
        }
    }
    return(argmax);
}
