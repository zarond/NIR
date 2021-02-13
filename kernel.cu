//nvcc -ptx "E:\семестр 7\НИР\kernel.cu" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64"
//nvcc -ptx "E:\семестр 7\НИР\kernel.cu" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64" -gencode arch=compute_35,code=sm_35 -rdc=true
#include "mex.h"
#include "gpu/mxGPUArray.h"
        

__global__ void kernel(float *U, const unsigned int N,const unsigned int M,const int t,const float v,const float d_x,const float d_t, const float b=20)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N*M) return;
    unsigned int x = i % M;
    unsigned int y = i / M;
    if (x+1>=M || y+1>=N || x<1 || y<1) {U[(x+y*M)*4+(t+1)%3]=0.0;return;}
	
    float val;
    if (U[(x+y*M)*4+3] > 0.0){
        val = -4*U[(x+y*M)*4+t%3];
        val+= (U[(x-1+y*M)*4+3]>0.0)? U[(x-1+y*M)*4+t%3]:0;//U[(x+1+y*M)*4+t%3];
        val+= (U[(x+1+y*M)*4+3]>0.0)? U[(x+1+y*M)*4+t%3]:0;//U[(x-1+y*M)*4+t%3];
        val+= (U[(x+M+y*M)*4+3]>0.0)? U[(x+M+y*M)*4+t%3]:0;//U[(x-M+y*M)*4+t%3];
        val+= (U[(x-M+y*M)*4+3]>0.0)? U[(x-M+y*M)*4+t%3]:0;//U[(x+M+y*M)*4+t%3];
        val*=(U[(x+y*M)*4+3]*U[(x+y*M)*4+3])*v*v*d_t*d_t/(d_x*d_x);  // ???
        //float b=20;
        val += 2*U[(x+y*M)*4+t%3]-U[(x+y*M)*4+(t-1)%3]*(1-d_t*b*0.5);
        val/=(1+d_t*b*0.5);
    } else {val = 0.0;}
    U[(x+y*M)*4+(t+1)%3]=val;
    //}

}

__global__ 
void kernelM(float *U, float* IR, const unsigned int N,const unsigned int M,const int t,
        const int T,const float v,const float d_x,const float d_t, const int x_ir,const int y_ir)
{
    dim3 blockDim(1024, 1, 1);
    dim3 gridDim(N*M/1024, 1, 1);
    for (int time=t;time<T;++time){
        kernel<<<gridDim,blockDim>>>(U,N,M,time+1,v,d_x,d_t);
        //__syncthreads();
        cudaDeviceSynchronize();
        IR[time]=U[(x_ir+y_ir*M)*4+(time+2)%3];
    }
}

__global__ void kernelAndSetIR(float *U, float *IR, const unsigned int N,const unsigned int M,const int t,const float v,const float d_x,const float d_t,const int x_ir,const int y_ir, const float b=20)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N*M) return;
    unsigned int x = i % M;
    unsigned int y = i / M;
    if (x+1>=M || y+1>=N || x<1 || y<1) {U[(x+y*M)*4+(t+1)%3]=0.0;return;}
	
    float val;
    if (U[(x+y*M)*4+3] > 0.0){
        val = -4*U[(x+y*M)*4+t%3];
        val+= (U[(x-1+y*M)*4+3]>0.0)? U[(x-1+y*M)*4+t%3]:0;//U[(x+1+y*M)*4+t%3];
        val+= (U[(x+1+y*M)*4+3]>0.0)? U[(x+1+y*M)*4+t%3]:0;//U[(x-1+y*M)*4+t%3];
        val+= (U[(x+M+y*M)*4+3]>0.0)? U[(x+M+y*M)*4+t%3]:0;//U[(x-M+y*M)*4+t%3];
        val+= (U[(x-M+y*M)*4+3]>0.0)? U[(x-M+y*M)*4+t%3]:0;//U[(x+M+y*M)*4+t%3];
        val*=(U[(x+y*M)*4+3]*U[(x+y*M)*4+3])*v*v*d_t*d_t/(d_x*d_x); // ???
        //float b=20;
        val += 2*U[(x+y*M)*4+t%3]-U[(x+y*M)*4+(t-1)%3]*(1-d_t*b*0.5);
        val/=(1+d_t*b*0.5);
    } else {val = 0.0;}
    U[(x+y*M)*4+(t+1)%3]=val;
    if (x==x_ir && y==y_ir) IR[t-1]=val; // (t-1) because we run with t+1
    //}

}

__global__ void kernelAndSetIRAndSource(float *U, float *IR, const unsigned int N,const unsigned int M,const int t,const float v,const float d_x,const float d_t,const int x_ir,const int y_ir, const int x_s,const int y_s,float* F,const float b=20)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N*M) return;
    unsigned int x = i % M;
    unsigned int y = i / M;
    if (x+1>=M || y+1>=N || x<1 || y<1) {U[(x+y*M)*4+(t+1)%3]=0.0;return;}
	
    float val;
    if (U[(x+y*M)*4+3] > 0.0){
        val = -4*U[(x+y*M)*4+t%3];
        val+= (U[(x-1+y*M)*4+3]>0.0)? U[(x-1+y*M)*4+t%3]:0;//U[(x+1+y*M)*4+t%3];
        val+= (U[(x+1+y*M)*4+3]>0.0)? U[(x+1+y*M)*4+t%3]:0;//U[(x-1+y*M)*4+t%3];
        val+= (U[(x+M+y*M)*4+3]>0.0)? U[(x+M+y*M)*4+t%3]:0;//U[(x-M+y*M)*4+t%3];
        val+= (U[(x-M+y*M)*4+3]>0.0)? U[(x-M+y*M)*4+t%3]:0;//U[(x+M+y*M)*4+t%3];
        if (x==x_s && y==y_s) val+=F[t-1]; // Source
        val*=(U[(x+y*M)*4+3]*U[(x+y*M)*4+3])*v*v*d_t*d_t/(d_x*d_x); // ???
        //float b=20;
        val += 2*U[(x+y*M)*4+t%3]-U[(x+y*M)*4+(t-1)%3]*(1-d_t*b*0.5);
        val/=(1+d_t*b*0.5);
    } else {val = 0.0;}
    U[(x+y*M)*4+(t+1)%3]=val;
    if (x==x_ir && y==y_ir) IR[t-1]=val; // (t-1) because we run with t+1
    //}

}

        
void mexFunction(int n_out, mxArray *Arr_out[], int n_in, const mxArray *Arr_in[])
{
    char const * const errId = "parallel:gpu:kernelM:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";
    /* Throw an error if the input is not a GPU array. */
    if ((n_out!=2) || (n_in<11) || !(mxIsGPUArray(Arr_in[0])) || !(mxIsGPUArray(Arr_in[1]))) {
       mexErrMsgIdAndTxt(errId, errMsg);
    }

    /* Declare all variables.*/
    mxGPUArray *Ugpu;
    mxGPUArray *IRgpu;
    mxGPUArray *Fgpu;
    float *d_U;
    float *d_IR;
    float *d_F;

    //INPUTS
    const unsigned int N=mxGetScalar(Arr_in[2]);
    const unsigned int M=mxGetScalar(Arr_in[3]);
    const int t=mxGetScalar(Arr_in[4]);
    const int T=mxGetScalar(Arr_in[5]);
    const float v=(float)*mxGetPr(Arr_in[6]);
    const float d_x=(float)*mxGetPr(Arr_in[7]);
    const float d_t=(float)*mxGetPr(Arr_in[8]);
    const int x_ir=mxGetScalar(Arr_in[9]);
    const int y_ir=mxGetScalar(Arr_in[10]);
    int mode,x_s,y_s;
    float b;
    if (n_in>=12) mode=mxGetScalar(Arr_in[11]);
    else mode=1;
    if (n_in>=13) b=(float)*mxGetPr(Arr_in[12]);
    if (n_in>=16) {
        x_s = mxGetScalar(Arr_in[13]);
        y_s = mxGetScalar(Arr_in[14]);
    }
           

    mxInitGPU();

    
    Ugpu = mxGPUCopyFromMxArray(Arr_in[0]);
    IRgpu = mxGPUCopyFromMxArray(Arr_in[1]);
    if (n_in>=16) {
        Fgpu = mxGPUCopyFromMxArray(Arr_in[15]);
        d_F = (float *)(mxGPUGetData(Fgpu));
    }

    if ((mxGPUGetClassID(Ugpu) != mxSINGLE_CLASS) || (mxGPUGetClassID(IRgpu) != mxSINGLE_CLASS)) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    d_U = (float *)(mxGPUGetData(Ugpu));
    d_IR = (float *)(mxGPUGetData(IRgpu));

    if (mode==0){
        kernelM<<<1, 1>>>(d_U, d_IR, N, M, t, T, v, d_x, d_t,x_ir,y_ir);
    } 
    else if (mode==1) {
        dim3 blockDim(1024, 1, 1);
        dim3 gridDim(N*M/1024, 1, 1);
        for (int time=t;time<T;++time){
            kernelAndSetIR<<<gridDim,blockDim>>>(d_U,d_IR,N,M,time+1,v,d_x,d_t,x_ir,y_ir,b);
            cudaDeviceSynchronize();
        }
    } else if (mode==2 || mode==3){
        dim3 blockDim(1024, 1, 1);
        dim3 gridDim(N*M/1024, 1, 1);
        for (int time=t;time<T;++time){
            kernelAndSetIRAndSource<<<gridDim,blockDim>>>(d_U,d_IR,N,M,time+1,v,d_x,d_t,x_ir,y_ir,x_s,y_s,d_F,b);
            cudaDeviceSynchronize();
        }
    }

    cudaDeviceSynchronize();

    /* Wrap the result up as a MATLAB gpuArray for return. */
    Arr_out[0] = mxGPUCreateMxArrayOnGPU(Ugpu);
    Arr_out[1] = mxGPUCreateMxArrayOnGPU(IRgpu);

    if (n_in>=16) {
        mxGPUDestroyGPUArray(Fgpu);
    }
}
