/*
 * projected_gradient_method.h
 *
 *  Created on: Oct 21, 2019
 *      Author: idris
 */

#ifndef PROJECTED_GRADIENT_METHOD_H_
#define PROJECTED_GRADIENT_METHOD_H_

/* Parameters */
//#define DOUBLE_PRECISION
#ifdef DOUBLE_PRECISION
typedef double pgm_float;
#define PGM_EPS_ABS (1e-3)
#define PGM_EPS_REL (1e-3)
#else
typedef float pgm_float;
#define PGM_EPS_ABS (1e-3f)
#define PGM_EPS_REL (1e-3f)
#endif
#define PGM_DIM (10)
#define PGM_MAX_ITER (1000)
#define PGM_CHECK_TERMINATION (60)

/* Functions */

/*
 * PGM_initialize:
 * Copies problem data and assigns the projection function: void proj_func(const float * restrict in, float * restrict out).
 * Needs to be called before PGM_solve.
 */
void PGM_initialize(const pgm_float * obj_func_matrix, const pgm_float * obj_func_vector,
					const pgm_float obj_func_grad_lipschitz_const, void (*proj_func) (const pgm_float * restrict in, pgm_float * restrict out));

/*
 * PGM_solve:
 * Solves the QP. Warm-start using out-array if warm_start == 1. Solution in out. Returns 0
 * if problem has been solved. Returns 1 if maximum iterations reached.
 */
int PGM_solve(pgm_float * out, const int warm_start);
int PGM_get_num_iter(void);

/* Only here for tests */
void PGM_gradient_step(const pgm_float * restrict in, pgm_float * restrict out);

#endif /* PROJECTED_GRADIENT_METHOD_H_ */
