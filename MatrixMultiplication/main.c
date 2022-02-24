#define CL_TARGET_OPENCL_VERSION 300

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <CL/cl.h>

void mat_mul_cpu(int *A, int *B, int *C, int aH, int aW)
{

    for (size_t i = 0; i < aH; i++)
    {
        for (size_t j = 0; j < aW; j++)
        {

            int value = 0;

            for (int k = 0; k < aH; k++)
            {
                value += B[k * aH + j] * A[i * aW + k];
            }

            C[i * aH + j] = value;
        }
    }
}

int main()
{
    // read kernel code
    int k_fd = open("kernel.cl", O_RDONLY);
    struct stat s;
    fstat(k_fd, &s);

    char *kernelSource = (char *)malloc(s.st_size * sizeof(char));
    kernelSource = (char *)mmap(0, s.st_size, PROT_READ, MAP_PRIVATE, k_fd, 0);

    printf("%s\n", kernelSource);

    int r = 1024;
    int c = 1024;
    const size_t elements = r * c;
    const size_t datasize = sizeof(int) * elements;

    int *A = (int*)malloc(elements * sizeof(int));
    int *B = (int*)malloc(elements * sizeof(int));
    int *C = (int*)malloc(elements * sizeof(int));
    int *C_CPU = (int*)malloc(elements * sizeof(int));

    for(int i = 0; i < elements; i++){
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    // compute result on CPU
    mat_mul_cpu(A, B, C_CPU, r, c);


    
    char buffer[1024];
    cl_uint platforms;
    cl_int status;

    status = clGetPlatformIDs(0, NULL, &platforms);
    assert(status == CL_SUCCESS);
    assert(platforms >= 1);

    cl_platform_id *platform_ids;
    platform_ids = (cl_platform_id *)malloc(sizeof(cl_platform_id) * platforms);
    status = clGetPlatformIDs(platforms, platform_ids, NULL);
    assert(status == CL_SUCCESS);

    for (int i = 0; i < platforms; i++)
    {
        printf("----\n");
        clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, 1024, &buffer, NULL);
        printf("Platform = %s\n", buffer);
        clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VERSION, 1024, &buffer, NULL);
        printf("Platform Version = %s\n", buffer);
        printf("----\n");
    }


    // select GPU from platform
    cl_device_id device = NULL;
    for (size_t i = 0; i < platforms; i++)
    {
        status = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (status == CL_SUCCESS)
        {
            clGetDeviceInfo(device, CL_DEVICE_NAME, 1024, &buffer, NULL);
            printf("Device = %s\n", buffer);
            printf("----\n");
            break;
        }
    }

    assert(device != NULL);

    // get device context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    assert(status == CL_SUCCESS);

    // create command Queue. wee feed data into it down further below!
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &status);
    assert(status == CL_SUCCESS);

    // device side allocation
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize, NULL, &status);

    // write data to device buffer.
    clEnqueueWriteBuffer(queue, bufA, CL_FALSE, 0, datasize, A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufB, CL_FALSE, 0, datasize, B, 0, NULL, NULL);

    // create & build program from kernel string.
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &status);
    assert(status == CL_SUCCESS);

    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (status != CL_SUCCESS)
    {
        size_t buflen;
        char *buf;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &buflen);
        buf = (char *)malloc(buflen);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, buflen, buf, NULL);
        printf("%s\n", buf);
        free(buf);
        exit(1);
    }

    // create kernel. 
    // Note: kernel_name must match function name
    cl_kernel kernel = clCreateKernel(program, "mat_mul", &status);

    // set arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), (void *)&r);
    clSetKernelArg(kernel, 4, sizeof(int), (void *)&c);

    size_t globalWorkSize[2], localWorkSize[2];
    /* == clinfo output for GTX 1050Ti ==
    Max work item dimensions                        3
    Max work item sizes                             1024x1024x64
    Max work group size                             1024
    Preferred work group size multiple (device)     32
    Preferred work group size multiple (kernel)     32
    Warp size (NV)                                  32
    */
    globalWorkSize[0] = r;
    globalWorkSize[1] = c;
    localWorkSize[0] = 16;
    localWorkSize[1] = 16;

    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    
    // read back result
    status = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, datasize, C, 0, NULL, NULL);

    for (size_t i = 0; i < elements; i++)
    {
        assert(C_CPU[i] == C[i]);
    }

    printf("%s\n", "Results Verified!");

    //free kernel string
    munmap(kernelSource, s.st_size);
    close(k_fd);
    
    // free host memory
    free(A);
    free(B);
    free(C);

    // free OpenCL related resources
    clReleaseDevice(device);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseKernel(kernel);

    // free device memory
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);

    return 0;
}