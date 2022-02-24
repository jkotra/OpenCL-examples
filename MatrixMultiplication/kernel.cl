__kernel
void mat_mul(__global int *A, __global int *B, __global int *C, int M, int N){

    int tx = get_global_id(0);
    int ty = get_global_id(1);

    int value = 0;
    for(int k = 0; k < M; k++){
        value += A[ty * M + k] * B[k * N + tx];
    }

    C[ty * M + tx] = value;

}