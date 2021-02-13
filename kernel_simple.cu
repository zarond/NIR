//nvcc -ptx "E:\семестр 7\НИР\kernel.cu" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64"
//nvcc -ptx "E:\семестр 7\НИР\kernel.cu" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64" -gencode arch=compute_35,code=sm_35 -rdc=true
__device__ float getneighbor(float3 *A, const unsigned int N,const unsigned int M, const unsigned int x, const unsigned int y){
    if (A[x+y*M].z == 0.0) {
        return A[x+y*M].x;
    }
    return 0.0;//A[x+y*M].x;
}
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
        val*=(U[(x+y*M)*4+3]*U[(x+y*M)*4+3])*v*v*d_t*d_t/(d_x*d_x); // ???
        //float b=20;
        val += 2*U[(x+y*M)*4+t%3]-U[(x+y*M)*4+(t-1)%3]*(1-d_t*b*0.5);
        val/=(1+d_t*b*0.5);
    } else {val = 0.0;}
    U[(x+y*M)*4+(t+1)%3]=val;
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
}

__global__ void kernel2(float *U, float *U1, const unsigned int N,const unsigned int M,const float v,const float d_x,const float d_t)
{
    float3 *A=(float3*) U;
    float3 *A1=(float3*) U1;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N*M) return;
    unsigned int x = i % M;
    unsigned int y = i / M;
    if (x+1>=M || y+1>=N || x<1 || y<1) {    A1[x+y*M] = A[x+y*M]; return;}
	//A[i].x = i;
    float pos = A[x+y*M].x;
    float vel = A[x+y*M].y;
    float wall = A[x+y*M].z;
    if (wall == 0.0){
        float m = A[x+1+y*M].x;//getneighbor(A,N,M,x+1,y);
        m += A[x-1+y*M].x;//getneighbor(A,N,M,x-1,y);
        m += A[x+y*M+M].x;//getneighbor(A,N,M,x,y+1);
        m += A[x+y*M-M].x;//getneighbor(A,N,M,x,y-1);
        m *= .25;
        vel = 1.5*(1.0-wall)*(m-pos)+vel;
        pos=pos+vel;
    } else {pos = 0.0;vel = 0.0;}
    A1[x+y*M].x = pos;//+0.1;
    A1[x+y*M].y = vel;
    A1[x+y*M].z = wall;
}

// __global__ void kernel2(float *U, const unsigned int N,const unsigned int M,int t,const float v,const float d_x,const float d_t)
// {
//     float3 *A1=(float3*) U;
//     float3 *A2=(float3*) (U+1);
//     float3 *A3=(float3*) (U+2);
//     float3 *wall=(float3*) (U+3);
//     float3* A[3] = {A1,A2,A3};
// 	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (i >= N*M) return;
//     unsigned int x = i % M;
//     unsigned int y = i / M;
//     if (x+1>=M || y+1>=N || x<1 || y<1) {return;}
// 
//     //for (t:=0;t<1000;++t){
//     float val;
//     if (wall[x+y*M].x == 0.0){
//         val = A[t%3][x-1+y*M].x-4*A[t%3][x+y*M].x+A[t%3][x+1+y*M].x+A[t%3][x+y*M+M].x+A[t%3][x+y*M-M].x;
//         val*=v*v*d_t*d_t/(d_x*d_x);
//         val += 2*A[t%3][x+y*M].x-A[(t-1)%3][x+y*M].x;
//     } else {val = 0.0;}
//     A[(t+1)%3][x+y*M].x=val;
// 
// }

