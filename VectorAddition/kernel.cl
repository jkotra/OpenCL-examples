__kernel 
void vec_add(__global int *A, __global int *B, __global int *C) {
  size_t idx = get_global_id(0);
  C[idx] = A[idx] + B[idx];
}