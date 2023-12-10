/*
 * AD tool to FORCESPRO Template - missing information to be filled in by createADTool.m 
 * (C) embotech AG, Zurich, Switzerland, 2013-2023. All rights reserved.
 *
 * This file is part of the FORCESPRO client, and carries the same license.
 */ 

#ifdef __cplusplus
extern "C" {
#endif

#include "include/gp_mpcc_h2h_ego.h"

#ifndef NULL
#define NULL ((void *) 0)
#endif

#include "gp_mpcc_h2h_ego_model.h"



/* copies data from sparse matrix into a dense one */
static void gp_mpcc_h2h_ego_sparse2fullcopy(solver_int32_default nrow, solver_int32_default ncol, const solver_int32_default *colidx, const solver_int32_default *row, gp_mpcc_h2h_ego_callback_float *data, gp_mpcc_h2h_ego_float *out)
{
    solver_int32_default i, j;
    
    /* copy data into dense matrix */
    for(i=0; i<ncol; i++)
    {
        for(j=colidx[i]; j<colidx[i+1]; j++)
        {
            out[i*nrow + row[j]] = ((gp_mpcc_h2h_ego_float) data[j]);
        }
    }
}




/* AD tool to FORCESPRO interface */
extern solver_int32_default gp_mpcc_h2h_ego_adtool2forces(gp_mpcc_h2h_ego_float *x,        /* primal vars                                         */
                                 gp_mpcc_h2h_ego_float *y,        /* eq. constraint multiplers                           */
                                 gp_mpcc_h2h_ego_float *l,        /* ineq. constraint multipliers                        */
                                 gp_mpcc_h2h_ego_float *p,        /* parameters                                          */
                                 gp_mpcc_h2h_ego_float *f,        /* objective function (scalar)                         */
                                 gp_mpcc_h2h_ego_float *nabla_f,  /* gradient of objective function                      */
                                 gp_mpcc_h2h_ego_float *c,        /* dynamics                                            */
                                 gp_mpcc_h2h_ego_float *nabla_c,  /* Jacobian of the dynamics (column major)             */
                                 gp_mpcc_h2h_ego_float *h,        /* inequality constraints                              */
                                 gp_mpcc_h2h_ego_float *nabla_h,  /* Jacobian of inequality constraints (column major)   */
                                 gp_mpcc_h2h_ego_float *hess,     /* Hessian (column major)                              */
                                 solver_int32_default stage,     /* stage number (0 indexed)                           */
                                 solver_int32_default iteration, /* iteration number of solver                         */
                                 solver_int32_default threadID   /* Id of caller thread                                */)
{
    /* AD tool input and output arrays */
    const gp_mpcc_h2h_ego_callback_float *in[4];
    gp_mpcc_h2h_ego_callback_float *out[7];
    

    /* Allocate working arrays for AD tool */
    
    gp_mpcc_h2h_ego_callback_float w[2031];
	
    /* temporary storage for AD tool sparse output */
    gp_mpcc_h2h_ego_callback_float this_f = (gp_mpcc_h2h_ego_callback_float) 0.0;
    gp_mpcc_h2h_ego_float nabla_f_sparse[11];
    gp_mpcc_h2h_ego_float h_sparse[9];
    gp_mpcc_h2h_ego_float nabla_h_sparse[26];
    gp_mpcc_h2h_ego_float c_sparse[14];
    gp_mpcc_h2h_ego_float nabla_c_sparse[65];
    
    
    /* pointers to row and column info for 
     * column compressed format used by AD tool */
    solver_int32_default nrow, ncol;
    const solver_int32_default *colind, *row;
    
    /* set inputs for AD tool */
    in[0] = x;
    in[1] = p;
    in[2] = l;
    in[3] = y;

	if ((0 <= stage && stage <= 8))
	{
		
		
		out[0] = &this_f;
		out[1] = nabla_f_sparse;
		gp_mpcc_h2h_ego_objective_0(in, out, NULL, w, 0);
		if( nabla_f != NULL )
		{
			nrow = gp_mpcc_h2h_ego_objective_0_sparsity_out(1)[0];
			ncol = gp_mpcc_h2h_ego_objective_0_sparsity_out(1)[1];
			colind = gp_mpcc_h2h_ego_objective_0_sparsity_out(1) + 2;
			row = gp_mpcc_h2h_ego_objective_0_sparsity_out(1) + 2 + (ncol + 1);
				
			gp_mpcc_h2h_ego_sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		}
		
		out[0] = c_sparse;
		out[1] = nabla_c_sparse;
		gp_mpcc_h2h_ego_dynamics_0(in, out, NULL, w, 0);
		if( c != NULL )
		{
			nrow = gp_mpcc_h2h_ego_dynamics_0_sparsity_out(0)[0];
			ncol = gp_mpcc_h2h_ego_dynamics_0_sparsity_out(0)[1];
			colind = gp_mpcc_h2h_ego_dynamics_0_sparsity_out(0) + 2;
			row = gp_mpcc_h2h_ego_dynamics_0_sparsity_out(0) + 2 + (ncol + 1);
				
			gp_mpcc_h2h_ego_sparse2fullcopy(nrow, ncol, colind, row, c_sparse, c);
		}
		if( nabla_c != NULL )
		{
			nrow = gp_mpcc_h2h_ego_dynamics_0_sparsity_out(1)[0];
			ncol = gp_mpcc_h2h_ego_dynamics_0_sparsity_out(1)[1];
			colind = gp_mpcc_h2h_ego_dynamics_0_sparsity_out(1) + 2;
			row = gp_mpcc_h2h_ego_dynamics_0_sparsity_out(1) + 2 + (ncol + 1);
				
			gp_mpcc_h2h_ego_sparse2fullcopy(nrow, ncol, colind, row, nabla_c_sparse, nabla_c);
		}
		
		out[0] = h_sparse;
		out[1] = nabla_h_sparse;
		gp_mpcc_h2h_ego_inequalities_0(in, out, NULL, w, 0);
		if( h != NULL )
		{
			nrow = gp_mpcc_h2h_ego_inequalities_0_sparsity_out(0)[0];
			ncol = gp_mpcc_h2h_ego_inequalities_0_sparsity_out(0)[1];
			colind = gp_mpcc_h2h_ego_inequalities_0_sparsity_out(0) + 2;
			row = gp_mpcc_h2h_ego_inequalities_0_sparsity_out(0) + 2 + (ncol + 1);
				
			gp_mpcc_h2h_ego_sparse2fullcopy(nrow, ncol, colind, row, h_sparse, h);
		}
		if( nabla_h != NULL )
		{
			nrow = gp_mpcc_h2h_ego_inequalities_0_sparsity_out(1)[0];
			ncol = gp_mpcc_h2h_ego_inequalities_0_sparsity_out(1)[1];
			colind = gp_mpcc_h2h_ego_inequalities_0_sparsity_out(1) + 2;
			row = gp_mpcc_h2h_ego_inequalities_0_sparsity_out(1) + 2 + (ncol + 1);
				
			gp_mpcc_h2h_ego_sparse2fullcopy(nrow, ncol, colind, row, nabla_h_sparse, nabla_h);
		}
	}
	if ((9 == stage))
	{
		
		
		out[0] = &this_f;
		out[1] = nabla_f_sparse;
		gp_mpcc_h2h_ego_objective_1(in, out, NULL, w, 0);
		if( nabla_f != NULL )
		{
			nrow = gp_mpcc_h2h_ego_objective_1_sparsity_out(1)[0];
			ncol = gp_mpcc_h2h_ego_objective_1_sparsity_out(1)[1];
			colind = gp_mpcc_h2h_ego_objective_1_sparsity_out(1) + 2;
			row = gp_mpcc_h2h_ego_objective_1_sparsity_out(1) + 2 + (ncol + 1);
				
			gp_mpcc_h2h_ego_sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		}
		
		out[0] = h_sparse;
		out[1] = nabla_h_sparse;
		gp_mpcc_h2h_ego_inequalities_1(in, out, NULL, w, 0);
		if( h != NULL )
		{
			nrow = gp_mpcc_h2h_ego_inequalities_1_sparsity_out(0)[0];
			ncol = gp_mpcc_h2h_ego_inequalities_1_sparsity_out(0)[1];
			colind = gp_mpcc_h2h_ego_inequalities_1_sparsity_out(0) + 2;
			row = gp_mpcc_h2h_ego_inequalities_1_sparsity_out(0) + 2 + (ncol + 1);
				
			gp_mpcc_h2h_ego_sparse2fullcopy(nrow, ncol, colind, row, h_sparse, h);
		}
		if( nabla_h != NULL )
		{
			nrow = gp_mpcc_h2h_ego_inequalities_1_sparsity_out(1)[0];
			ncol = gp_mpcc_h2h_ego_inequalities_1_sparsity_out(1)[1];
			colind = gp_mpcc_h2h_ego_inequalities_1_sparsity_out(1) + 2;
			row = gp_mpcc_h2h_ego_inequalities_1_sparsity_out(1) + 2 + (ncol + 1);
				
			gp_mpcc_h2h_ego_sparse2fullcopy(nrow, ncol, colind, row, nabla_h_sparse, nabla_h);
		}
	}
    
    /* add to objective */
    if (f != NULL)
    {
        *f += ((gp_mpcc_h2h_ego_float) this_f);
    }

    return 0;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
