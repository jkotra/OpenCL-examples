#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

int main()
{

    const size_t elements = 4096;
    const size_t datasize = sizeof(int) * elements;

    int *A = new int[elements];
    int *B = new int[elements];
    int *C = new int[elements];

    for (int i = 0; i < elements; i++)
    {
        A[i] = i;
        B[i] = i;
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    cl::Device device;

    for (auto platform : platforms)
    {
        std::vector<cl::Device> devices;
        std::cout << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::cout << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;

        try // try to get GPU device.
        {
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices); // might throw exception.
            for (auto _device : devices)
            {
                std::cout << _device.getInfo<CL_DEVICE_NAME>() << std::endl;
                device = _device;
                break;
            }
            break;
        }
        catch (cl::Error err)
        {
            continue;
        }
    }

    cl::Context ctx(device);

    cl::CommandQueue queue = cl::CommandQueue(ctx, device);

    cl::Buffer bufA = cl::Buffer(ctx, CL_MEM_READ_ONLY, datasize);
    queue.enqueueWriteBuffer(bufA, CL_TRUE, 0, datasize, A);
    cl::Buffer bufB = cl::Buffer(ctx, CL_MEM_READ_ONLY, datasize);
    queue.enqueueWriteBuffer(bufB, CL_TRUE, 0, datasize, B);
    cl::Buffer bufC = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, datasize);
    queue.enqueueWriteBuffer(bufC, CL_TRUE, 0, datasize, C);

    std::ifstream f("../kernel.cl", std::ios::in);
    std::string kernelSource;

    // load kernel string
    std::stringstream ss;
    ss << f.rdbuf();
    kernelSource = ss.str();
    std::cout << kernelSource << std::endl;

    cl::Program program(ctx, kernelSource.c_str());
    try
    {
        program.build();
    }
    catch (cl::Error err)
    {

        cl::BuildLogType build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();

        for (auto l : build_log)
        {
            cl::Device d = std::get<0>(l);
            cl::string err = std::get<1>(l);
            std::cout << "Error Exception on:" << d.getInfo<CL_DEVICE_NAME>() << "\n"
                      << err << std::endl;
            exit(1);
        }
    }

    cl::Kernel vec_add_kernel(program, "vec_add");
    vec_add_kernel.setArg(0, bufA);
    vec_add_kernel.setArg(1, bufB);
    vec_add_kernel.setArg(2, bufC);

    cl::NDRange global(elements);
    cl::NDRange local(32);

    queue.enqueueNDRangeKernel(vec_add_kernel, 0, global, local);

    queue.enqueueReadBuffer(bufC, CL_TRUE, 0, datasize, C);

    for (int i = 0; i < elements; i++)
    {
        std::cout << C[i] << "\t";
    }

    return 0;
}