#include <assert.h>
#include <stdio.h>

#include "projected_gradient_method.h"

/* Problem Data */
pgm_float PGM_in_mat[PGM_DIM * PGM_DIM];
pgm_float PGM_in_vec[PGM_DIM];
pgm_float PGM_dec_var[PGM_DIM];
pgm_float PGM_inv_lip_const;
int PGM_last_num_iter;
int PGM_is_initialized;
void (*PGM_project) (const pgm_float * restrict in, pgm_float * restrict out);
// we also make use of the out-array passed to solve()

/* Prototypes */
//void PGM_gradient_step(const pgm_float * restrict in, pgm_float * restrict out);
//void PGM_project(const pgm_float * restrict in, pgm_float * restrict out);

/* Prototypes algebra */
void PGM_vec_copy(const pgm_float * restrict in, pgm_float * restrict out, const int len);
void PGM_vec_swap(const pgm_float * in_out1, const pgm_float * in_out2);
void PGM_vec_init(pgm_float * out, const pgm_float in);
pgm_float PGM_max(pgm_float in1, pgm_float in2);
pgm_float PGM_abs_float(pgm_float in);
pgm_float PGM_inf_norm(const pgm_float * in);
pgm_float PGM_inf_norm_error(const pgm_float * in1, const pgm_float * in2);

int PGM_solve(pgm_float * out, const int warm_start) {
	int i_iter;
	pgm_float abs_error, last_iter_inf_norm;
	assert(PGM_is_initialized == 1);

	if (warm_start != 1)
		PGM_vec_init(out, 0.0);

	for (i_iter = 0; i_iter < PGM_MAX_ITER; i_iter ++) {
		/*
		 * Gradient step: y(k+1) = y(k) - 1/L * (J*y(k) + q)
		 * with y(k) = out, y(k+1) = PGM_dev_var, k = i_iter
		 */
		// PGM_gradient_step(const pgm_float * restrict in, pgm_float * restrict out)
		PGM_gradient_step(out, PGM_dec_var);

		/*
		 * Projection: yp(k+1) = P(y(k+1))
		 * with yp(k+1) = out, y(k+1) = PGM_dec_var, k = i_iter
		 */
		// void (*proj_func) (const pgm_float * restrict in, pgm_float * restrict out)
		PGM_project(PGM_dec_var, out);

		/*
		 * Check for termination.
		 */
		// TODO: that doesn't seem to work well, ends too early!
		if ((PGM_CHECK_TERMINATION) && (i_iter > 0) && (i_iter % PGM_CHECK_TERMINATION == 0)) {
			abs_error = PGM_inf_norm_error(out, PGM_dec_var);
			if (abs_error == 0) {
				break;
			} else {
				last_iter_inf_norm = PGM_inf_norm(PGM_dec_var);
				if (last_iter_inf_norm == 0) {
					break;
				} else {
					if ((abs_error < PGM_EPS_ABS) && (abs_error < PGM_EPS_REL * last_iter_inf_norm)) {
						break;
					}
				}
			}
		}
	}
	PGM_last_num_iter = i_iter;
	if (i_iter == PGM_MAX_ITER) {
		return 1;
	} else {
		return 0;
	}
}

void PGM_gradient_step(const pgm_float * restrict in, pgm_float * restrict out) {
	int i_row, i_col;
	pgm_float row_res;
	pgm_float * mat_ptr;
	for (i_row = 0; i_row < PGM_DIM; i_row++) {
		row_res = 0.0;
		mat_ptr = &PGM_in_mat[i_row * PGM_DIM];
		for (i_col = 0; i_col < PGM_DIM; i_col++, mat_ptr++) {
			row_res += (*mat_ptr) * in[i_col];
		}
		out[i_row] = in[i_row] - PGM_inv_lip_const * (row_res + PGM_in_vec[i_row]);
	}
}

/*
 * Algebra and miscellaneous
 */
void PGM_initialize(const pgm_float * obj_func_matrix, const pgm_float * obj_func_vector,
					const pgm_float obj_func_grad_lipschitz_const, void (*proj_func) (const pgm_float * restrict in, pgm_float * restrict out)) {
	PGM_inv_lip_const = 1.0 / obj_func_grad_lipschitz_const;
	PGM_vec_copy(obj_func_vector, PGM_in_vec, PGM_DIM);
	PGM_vec_copy(obj_func_matrix, PGM_in_mat, PGM_DIM * PGM_DIM);
	PGM_project = proj_func;
	PGM_is_initialized = 1;
}

void PGM_vec_init(pgm_float * out, const pgm_float in) {
	int i;
	for (i = 0; i < PGM_DIM; i++)
		out[i] = in;
}

void PGM_vec_swap(const pgm_float * in_out1, const pgm_float * in_out2) {
	const pgm_float * tmp = in_out1;
	in_out1 = in_out2;
	in_out2 = tmp;
}

void PGM_vec_copy(const pgm_float * restrict in, pgm_float * restrict out, const int len) {
	int i;
	for (i = 0; i < len; i++) {
		out[i] = in[i];
	}
}

pgm_float PGM_max(pgm_float in1, pgm_float in2) {
	return (in1 > in2) ? in1 : in2;
}

pgm_float PGM_abs_float(pgm_float in) {
	return (in > 0) ? in : -in;
}

pgm_float PGM_inf_norm(const pgm_float * in) {
	int i;
	pgm_float max_val = PGM_abs_float(in[0]);
	for (i = 1; i < PGM_DIM; i++)
		max_val = PGM_max(PGM_abs_float(in[i]), max_val);
	return max_val;
}

pgm_float PGM_inf_norm_error(const pgm_float * in1, const pgm_float * in2) {
	int i;
	pgm_float max_val = PGM_abs_float(in1[0] - in2[0]);
	for (i = 1; i < PGM_DIM; i++)
		max_val = PGM_max(PGM_abs_float(in1[i] - in2[i]), max_val);
	return max_val;
}

int PGM_get_num_iter(void) {
	return PGM_last_num_iter;
}
