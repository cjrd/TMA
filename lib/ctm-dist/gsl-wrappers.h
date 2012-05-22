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

#ifndef GSL_WRAPPERS_H
#define GSL_WRAPPERS_H

// #include <gsl/gsl_check_range.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <math.h>

// #define MAXFLOAT 3.40282347e+38F

double safe_log(double);
double log_sum(double, double);
double vget(const gsl_vector*, int);
void vset(gsl_vector*, int, double);
void vinc(gsl_vector*, int, double);
double mget(const gsl_matrix* m, int, int);
void mset(gsl_matrix*, int, int, double) ;
void minc(gsl_matrix*, int, int, double) ;
void col_sum(gsl_matrix*, gsl_vector*);
void vprint(const gsl_vector*);
void mprint(const gsl_matrix*);
void scanf_vector(char*, gsl_vector*);
void scanf_matrix(char*, gsl_matrix*);
void printf_vector(char*, gsl_vector*);
void printf_matrix(char*, gsl_matrix*);
double log_det(gsl_matrix*);
void matrix_inverse(gsl_matrix*, gsl_matrix*);
void sym_eigen(gsl_matrix*, gsl_vector*, gsl_matrix*);
double sum(gsl_vector*);
double norm(gsl_vector *);
void vfprint(const gsl_vector * v, FILE * f);
int argmax(gsl_vector *v);
void center(gsl_vector* v);
void normalize(gsl_vector* v);

#endif
