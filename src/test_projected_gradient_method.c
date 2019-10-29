#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "projected_gradient_method.h"
#include "box_rate_projection.h"
#include "test_data.h"
#include "workspace.h"
#include "osqp.h"

/* Some timing utils */
typedef struct BCINV_TIMER {
#ifdef SOC_C6678
	unsigned long int tic;
	unsigned long int toc;
#else
	struct timespec tic;
	struct timespec toc;
#endif
	int n_calls;
	float t_cumsum;
} BCINVTimer;

BCINVTimer* construct_timer();
void init_timer(BCINVTimer *timer);
void destruct_timer(BCINVTimer *timer);
void bcinv_tic(BCINVTimer *t);
void bcinv_toc(BCINVTimer *t);
float average_time(BCINVTimer *t);
void print_timer_info(BCINVTimer timer, char *timer_name);

#define ABS_ERROR_TOL (1e-6f)

pgm_float pgm_solution[PGM_DIM];
pgm_float test_sol[10];

void test_gradient_step() {
	printf("Testing PGM_gradient_step()\n");
	int i;
	pgm_float error = 0.0;
	PGM_initialize(obj_func_mat, obj_func_vec, obj_func_lipsch_const, symmetric_box_rate_projection);
	PGM_gradient_step(test_input_gradient_step, test_sol);
	for (i = 0; i < PGM_DIM; i++) {
		error += (test_output_gradient_step[i] - test_sol[i]) * (test_output_gradient_step[i] - test_sol[i]);
	}
	printf("error = %f\n", error);
	assert(error < ABS_ERROR_TOL);
	printf("Test PGM_gradient_step() passed\n");
}

void test_algorithm() {
	printf("Testing algorithm()\n");
	int retval, i;
	pgm_float error = 0.0;
	pgm_float error2 = 0.0;
	BCINVTimer * osqp_timer = construct_timer();
	BCINVTimer * pgm_timer = construct_timer();

	// Get OSQP solution: requires linking with OSQP using floats and ints
	bcinv_tic(osqp_timer);
	osqp_solve(&workspace);
	bcinv_toc(osqp_timer);

	// Get PGM solution: requires linking with box_rate_projection
	PGM_initialize(obj_func_mat, obj_func_vec, obj_func_lipsch_const, symmetric_box_rate_projection);
	bcinv_tic(pgm_timer);
	retval = PGM_solve(pgm_solution, 0);
	bcinv_toc(pgm_timer);

	for (i = 0; i < PGM_DIM; i++) {
		printf("sol_osqp_matl[%d] = %.4f, sol_pgm[%d] = %.4f, sol_osqp_C[%d] = %.4f\n",
				i, osqp_sol[i], i, pgm_solution[i], i, workspace.solution->x[i]);
		error += (osqp_sol[i]- pgm_solution[i]) * (osqp_sol[i]- pgm_solution[i]);
		error2 += ((pgm_float)workspace.solution->x[i]- pgm_solution[i]) * ((pgm_float)workspace.solution->x[i]- pgm_solution[i]);
	}

	printf("Return value = %d (0 = success)\n", retval);
	printf("Num iter OSQP = %d, Num iter PGM = %d, Num iter OSQP C = %d\n", num_iter_osqp, PGM_get_num_iter(), workspace.info->iter);
	printf("Time per iter OSQP = %.3f mus, Time per iter PGM = %.3f mus\n",
			average_time(osqp_timer) / ((float)workspace.info->iter) * 1000000.0,
			average_time(pgm_timer) / ((float)PGM_get_num_iter()) * 1000000.0);
	printf("Error OSQP matl = %.6f, Error OSQP C = %.6f\n", error, error2);
}

int main() {
	// NOTE: need to link with box_rate_projection and assert for equal dimension
	assert(BRP_DIM == PGM_DIM);

	test_gradient_step();
	test_algorithm();

	return 0;
}



/* Utils */
BCINVTimer* construct_timer() {
	BCINVTimer *timer = (BCINVTimer *)malloc(sizeof(BCINVTimer));
	init_timer(timer);
	return timer;
}

void destruct_timer(BCINVTimer *timer) {
	free(timer);
}

void init_timer(BCINVTimer *timer) {
	timer->t_cumsum = 0.0;
	timer->n_calls = 0;
}

void bcinv_tic(BCINVTimer *t) {
	clock_gettime(CLOCK_MONOTONIC, &t->tic);
}

void bcinv_toc(BCINVTimer *t) {
  struct timespec temp;

  clock_gettime(CLOCK_MONOTONIC, &t->toc);

  if ((t->toc.tv_nsec - t->tic.tv_nsec) < 0) {
    temp.tv_sec  = t->toc.tv_sec - t->tic.tv_sec - 1;
    temp.tv_nsec = 1e9 + t->toc.tv_nsec - t->tic.tv_nsec;
  } else {
    temp.tv_sec  = t->toc.tv_sec - t->tic.tv_sec;
    temp.tv_nsec = t->toc.tv_nsec - t->tic.tv_nsec;
  }

  t->t_cumsum += (float)temp.tv_sec + (float)temp.tv_nsec / 1e9;
  t->n_calls += 1;
}

float average_time(BCINVTimer *t) {
	return t->t_cumsum / ((float)t->n_calls);
}

void print_timer_info(BCINVTimer timer, char *timer_name) {
	const double avg_time_seconds = average_time(&timer);
	const long int ncycles = 0;
	printf("Time %s / n_calls: %.6f s e-6 (n_calls=%d) (Freq = %.6f Hz) (Cycles = %ld)\n",
			timer_name, avg_time_seconds*1000000.0, (int)timer.n_calls,
			1.0 / (avg_time_seconds), ncycles);
}

