/*=============================================================================
 * fast_tree_ids.c
 * 
 * This code performs simple decision tree evaluation for an array-based
 * representation of a decision tree defined for real-valued observations.
 * This particular bit of code only returns the 'node id' for each input.
 *
 * The tree is passed in as an array f_tree structured as follows:
 *   f_tree(i,1): the feature on which node i splits
 *   f_tree(i,2): the feature value at which node i splits
 *   f_tree(i,3): the 'node id' of the left child of node i
 *   f_tree(i,4): the 'node id' of the right child of node i
 *   f_tree(i,5): the tree depth of node i
 *   f_tree(i,6:end): the weights associated with node i
 *
 *   note: node i is the node with 'node id' i.
 *
 *===========================================================================*/

#include "mex.h"
#include "math.h"

#define EPS 0.000001

/* The computational routine */
void get_ids(x_obs, x_ids, f_tree, max_depth, obs_count, tree_rows, tree_cols)
    double *x_obs, *x_ids, *f_tree;
    int max_depth, obs_count, tree_rows, tree_cols;
{
    double thresh_val;
    int i, id, th_off, lc_off, rc_off, de_off, xf_off;
    th_off = tree_rows;
    lc_off = tree_rows * 2;
    rc_off = tree_rows * 3;
    de_off = tree_rows * 4;
    for (i=0; i<obs_count; i++) {
        id = 0;
        xf_off = (int) ((f_tree[id] - 1) * obs_count);
        while ((f_tree[lc_off + id] > 0.5) && (f_tree[de_off + id] < max_depth)) {
            if (x_obs[xf_off + i] <= f_tree[th_off + id]) {
                id = (int) (f_tree[lc_off + id] - 1);
            } else {
                id = (int) (f_tree[rc_off + id] - 1);
            }
            xf_off = (int) ((f_tree[id] - 1) * obs_count);
        }
        x_ids[i] = (double) (id + 1);
    }
    return;
}

/* The gateway function */
void mexFunction(nlhs, plhs, nrhs, prhs)
    int nlhs, nrhs;
    mxArray *plhs[];
    const mxArray *prhs[];
{
    double *x_obs, *x_ids, *f_tree;
    int max_depth, obs_count, tree_rows, tree_cols;

    /* check for proper number of arguments */
    if (nrhs != 3) {
        mexErrMsgTxt("fast_tree_ids(): requires three inputs.");
    }
    if (nlhs != 1) {
        mexErrMsgTxt("fast_tree_ids(): requires one output.");
    }

    /* Get the inputs */
    x_obs = mxGetPr(prhs[0]);
    f_tree = mxGetPr(prhs[1]);
    max_depth = (int) mxGetScalar(prhs[2]);

    /* Get the number of rows in the observation and tree arrays */
    obs_count = (int) mxGetM(prhs[0]);
    tree_rows = (int) mxGetM(prhs[1]);
    tree_cols = (int) mxGetN(prhs[1]);

    /* Init an array to hold the retrieved node ids */
    plhs[0] = mxCreateDoubleMatrix(obs_count,1,mxREAL);
    x_ids = mxGetPr(plhs[0]);

    /* Fill the id array for the given observations and tree */
    get_ids(x_obs, x_ids, f_tree, max_depth, obs_count, tree_rows, tree_cols);

    /* bam, average length computation */
    return;
}
