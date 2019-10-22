#include <assert.h>
#include <stdio.h>

#include "projected_gradient_method.h"
#include "box_rate_projection.h"
#include "test_data_1.h"

#define ABS_ERROR_TOL (1e-6f)

float pgm_solution[PGM_DIM];

float test_sol[10];
float test_array_gradient_step[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
float test_result_gradient_step[10] = {-2.5463, -0.9704, 0.9609, 3.2131, 6.2482, 5.4503, 2.5248, 4.6547, 6.7072, 5.8441};

void test_gradient_step() {
	printf("Testing PGM_gradient_step()\n");
	int i;
	float error = 0.0;
	PGM_initialize(obj_func_mat, obj_func_vec, obj_func_lipsch_const, symmetric_box_rate_projection);
	PGM_gradient_step(test_array_gradient_step, test_sol);
	for (i = 0; i < PGM_DIM; i++) {
		error += (test_result_gradient_step[i] - test_sol[i]) * (test_result_gradient_step[i] - test_sol[i]);
	}
	assert(error < ABS_ERROR_TOL);
	printf("Test PGM_gradient_step() passed\n");
}

void test_algorithm() {
	int retval, i;
	float error = 0.0;

	PGM_initialize(obj_func_mat, obj_func_vec, obj_func_lipsch_const, symmetric_box_rate_projection);
	retval = PGM_solve(pgm_solution, 0);

	for (i = 0; i < PGM_DIM; i++) {
		printf("sol_osqp[%d] = %.4f, sol_pgm[%d] = %.4f\n", i, osqp_sol[i], i, pgm_solution[i]);
		error += (osqp_sol[i]- pgm_solution[i]) * (osqp_sol[i]- pgm_solution[i]);
	}

	printf("Return value = %d (0 = success)\n", retval);
	printf("Num iter OSQP = %d, Num iter PGM = %d\n", num_iter_osqp, PGM_get_num_iter());
	printf("Error = %.6f\n", error);
}

int main() {
	test_gradient_step();
	test_algorithm();

	return 0;
}


