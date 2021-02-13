//nvcc -ptx "E:\семестр 7\НИР\kernel_IS.cu" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64"
//nvcc -ptx "E:\семестр 7\НИР\kernel_IS.cu" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64" -gencode arch=compute_35,code=sm_35 -rdc=true
#include "inc\helper_math.h"
#include "inc\helper_math.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"
        
#define eps 0.000001
#define pi 3.141592654
#define MAXCUTS 16
        
inline __device__ float2 xy(float4 a) {return make_float2(a.x,a.y);}
inline __device__ float2 zw(float4 a) {return make_float2(a.z,a.w);}
inline __device__ bool operator==(float4 a, float b) {return (a.x==b && a.y==b && a.z==b && a.w==b);}
inline __device__ bool operator<=(float2 a, float b) {return (a.x<=b && a.y<=b);}
inline __device__ float4 make_float4(float2 a, float2 b) {return make_float4(a.x,a.y,b.x,b.y);}
        
inline __device__ float cross(float2 a, float2 b){return a.x*b.y-a.y*b.x;}
inline __device__ float2 normal(float2 a){return make_float2(-a.y,a.x);}
inline __device__ float2 reflect(float2 a, float4 b){
    float2 n = normalize(zw(b)-xy(b));
    return xy(b)+(-a+xy(b))-2*dot(n,(-a+xy(b)))*n;
}

struct ISource{
    float2 pos;
    float timeoffset;
    float4 window;
    int edgeid;
    int parent_source;
    int reflections;
    int difractions;
};

__device__ ISource* Sources;
__device__ unsigned int NumberOfSources = 0;

__device__ float2 IntersectRayEdge(float2 ro, float2 direction,float4 edge){
    float2 SourceDirection = make_float2(ro.x - edge.x,ro.y - edge.y);
    float2 edgedir = make_float2(edge.z-edge.x,edge.w-edge.y);
    float2 v3 = normal(direction);//make_float2(-direction.y, direction.x);

    float dotv = dot(edgedir,v3);
    if (dot(normal(edgedir),SourceDirection) < eps || abs(dotv) < eps) // обратная сторона
        return make_float2(0.0,0.0); // возвращает неверную ориентацию

    float t1 = cross(edgedir,SourceDirection) / dotv;
    float t2 = dot(SourceDirection,v3) / dotv;

    //if (t1 >= 0.0 && (t2 >= 0.0 && t2 <= 1.0) && (t1<=minR)){}
    return make_float2(t1,t2);
}

__device__ float4 CutEdgeWithBeam(float2 ro, float4 window, float4 edge, int sourceid=-1){
    // вычисляем пересекает ли конус видимости ребро и не загорожено ли оно другими ребрами
    // сначала попадает ли ребро в конус видимости вообще
    bool CanBeSeen = (sourceid==0); // если id == 0 то значит - источник и распространяется во все стороны
    float2 t_1 = IntersectRayEdge(/*pos*/xy(window), xy(window) - ro, edge);
    float2 t_2 = IntersectRayEdge(/*pos*/zw(window), zw(window) - ro, edge);
    
    bool notseen =         
    !(t_1.x >= 0.0 && t_1.y >= 0.0 && t_1.y <= 1.0) && !(t_2.x >= 0.0 && t_2.y >= 0.0 && t_2.y <= 1.0) //ни одна, ни вторая не пересекают
    && !((t_2.y <= 0 && t_2.x >= 0 && (t_1.x >= 0 && t_1.y >= 1 || t_1.x <= 0 || t_1.x >= 0 && t_1.y <= t_2.y )) // и не один из этих случаев
    || (t_1.y >= 1 && t_1.x >= 0 && ( t_2.x <= 0 || t_2.x >= 0 && t_2.y >= t_1.y )) 
    || (t_1.x<=0 && t_2.x<=0 && t_2.y >= t_1.y))
    || (t_1.x==0 && t_2.x==0 && t_1.y==0 && t_2.y==0) || (dot(normal(zw(edge)-xy(edge)),ro-xy(edge)) < eps);

    CanBeSeen = CanBeSeen || !notseen;
    if (CanBeSeen==false) {
        return make_float4(0.0); // возврат, что не пересекается вообще //test
    }

    float4 newwindow = edge;
    float2 edgedir = make_float2(edge.z-edge.x,edge.w-edge.y);
    if (t_2.y>0 && t_2.x>0 && sourceid!=0) xy(newwindow) = xy(edge) + edgedir*t_2.y; // обрезание ребра конусом
    if (t_1.y<1 && t_1.x>0 && sourceid!=0) zw(newwindow) = xy(edge) + edgedir*t_1.y;
    
    return newwindow;
}

__device__ void CutBeamWithEdge(float2 ro, float4 window, float4 farwindow, float4 edge, float2* cuts, int &numberOfCuts){
    if (dot(normal(zw(farwindow) - xy(farwindow)),xy(edge)-xy(farwindow))<eps 
        && dot(normal(zw(farwindow) - xy(farwindow)),zw(edge)-xy(farwindow))<eps 
        || dot(edge,edge)<=eps)
        // ребро находится за farwindow (оба конца) или перед window (это отсеивается в cutEdgeWithBeam) или размер ребро - ноль
        return;
    // если подрезает с боков
    // записываем в cuts проекцию на farwindow (в процентах)
    float2 t1 = IntersectRayEdge(ro, zw(edge)-ro, farwindow);
    float2 t2 = IntersectRayEdge(ro, xy(edge)-ro, farwindow);
    //if (t1.x>0){ t1.y = max(t1.y,0.0); t1.y = min(t1.y,1.0);}
    if (t1.x>0){ t1.y = max(t1.y,0.0); t1.y = min(t1.y,1.0);}
    else if (t1.x<0) {return;t1.y = 1;}
    if (t2.x>0){ t2.y = max(t2.y,0.0); t2.y = min(t2.y,1.0);}
    //if (t2.x>0){ t2.y = max(t2.y,0.0); t2.y = min(t2.y,1.0);}
    else if (t2.x<0) {return;t2.y = 0;}

    if (t1.y - t2.y >= eps && t1.x!=0 && t2.x!=0) // не равны друг другу (можно было и t1.y != t2.y)
        cuts[numberOfCuts++] = make_float2(t2.y,t1.y); 
}

__device__ void sortCuts(float2* A,int N){
    for ( int i = 1;i < N;++i){
        float2 x = A[i];
        int j = i - 1;
        for (;j >= 0 && A[j].x > x.x;--j)
        //while (j >= 0 && A[j] > x)
        {
            A[j+1] = A[j];
            //j = j - 1;
        }
        A[j+1] = x;
    }
}

__device__ void AddNewSource(/*unsigned int &NumberOfSources, */const ISource source,float4 newwindow/*, ISource* Sources*/,unsigned int MaxSources, int edgeid, int sourceid){
    ISource newsource = {
        reflect(source.pos, newwindow),
        source.timeoffset,
        newwindow,
        edgeid,
        sourceid,
        source.reflections+1,
        source.difractions,
    };
    int idx = atomicAdd(&NumberOfSources, 1);
    if (idx<MaxSources)
        Sources[idx] = newsource;
}

__global__ void ComputeNewIS(const float4* Edges,/*ISource* Sources,*/ const unsigned int N, 
        const unsigned int MaxSources,const unsigned int CurrentNumberOfSources /*,unsigned int &CurrentSource, unsigned int &NumberOfSources*/){
    __shared__ float4 shared[1024]; // общая память под обрезание конусов ребрами
            
    int sourceid = blockIdx.x + CurrentNumberOfSources; //???
    int edgeid = threadIdx.x;
    float4 edge = Edges[edgeid];
    ISource source = Sources[sourceid];
    
    float4 newwindow = CutEdgeWithBeam(source.pos, source.window, edge, sourceid);
    shared[edgeid]=newwindow;
    __syncthreads();
    if (newwindow==0.0) {
        return; // возврат, что не пересекается вообще
    }
    //----------------------------------------------------------
    
    float2 cuts[MAXCUTS];// = new float2[MAXCUTS]; // максимум можно разрезать конус на MAXCUTS частей
    int numberOfCuts = 0;
    for (int i=0;i<N && numberOfCuts<MAXCUTS;++i){ // рассчитать заслон от других ребер
        edge = Edges[i];
        if (i==edgeid || i==source.edgeid) continue; // чтобы edge и window сами себя не отсекали ..? надо улучшить 
        CutBeamWithEdge(source.pos, source.window, newwindow, shared[i], cuts, numberOfCuts);
    }
    //numberOfCuts = 2; cuts[1]=make_float2(0.0,0.6);cuts[0]=make_float2(0.5,0.7);

    // отсортируем по cuts.x
    sortCuts(cuts,numberOfCuts);
    float2 lr=make_float2(0.0,0.0); // границы
        
    float* cutsSingle = (float*) cuts;
    int2 indexes = make_int2(0); 
    for (int i = 0/*2*i + ((cuts[i].x>0)? 0 : 1)*/;i<2*numberOfCuts;++i){        
        if (lr.y - lr.x > 0 && (indexes.x % 2 == 1 && indexes.y % 2 == 0 || indexes.x==0 && indexes.y==0)){
            // добавлять новые мнимые источники
            float4 cutwindow = make_float4(xy(newwindow)+lr.x*(zw(newwindow)-xy(newwindow)),xy(newwindow)+lr.y*(zw(newwindow)-xy(newwindow))); 
            AddNewSource(/*NumberOfSources,*/source,cutwindow/*,Sources*/,MaxSources,edgeid,sourceid);
        }
        lr.x = lr.y;
        indexes.x = indexes.y;
        if (cutsSingle[i] >= lr.y){
            lr.y =cutsSingle[i];
            indexes.y = i;
        }
    }
    if (lr.y<1){ //последний промежуток
        float4 cutwindow = make_float4(xy(newwindow)+lr.y*(zw(newwindow)-xy(newwindow)),zw(newwindow)); 
        AddNewSource(/*NumberOfSources,*/source,cutwindow/*,Sources*/,MaxSources,edgeid,sourceid);
    }

}

__global__ 
void kernelIS(const float4 * Edges, ISource* d_Sources, const unsigned int N, const unsigned int MaxSources, const float x_s,const float y_s)
{
    Sources = d_Sources;
    /*unsigned int */NumberOfSources = 1;
    unsigned int CurrentSource = 0;
    Sources[0]={make_float2(x_s,y_s),0.0,make_float4(0.0),-1,-1,0,0};
    
    for (;NumberOfSources<MaxSources;){
    	dim3 blockDim(N, 1, 1); // при условии что N<1024
        dim3 gridDim((NumberOfSources-CurrentSource), 1, 1);
        unsigned int tmp = NumberOfSources;
        ComputeNewIS<<<gridDim,blockDim>>>(Edges,/* Sources,*/ N, MaxSources, CurrentSource /*,CurrentSource, NumberOfSources*/);
        cudaDeviceSynchronize();
        CurrentSource = tmp;
    }
}

void mexFunction(int n_out, mxArray *Arr_out[], int n_in, const mxArray *Arr_in[])
{
    char const * const errId = "parallel:gpu:kernelM:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";
    /* Throw an error if the input is not a GPU array. */
    if ((n_out!=1) || (n_in<5) /*|| !(mxIsGPUArray(Arr_in[0]))*/ /*|| !(mxIsGPUArray(Arr_in[3]))*/) {
       mexErrMsgIdAndTxt(errId, errMsg);
    }

    //INPUTS
    const unsigned int N = mxGetScalar(Arr_in[1]);
    const unsigned int MaxSources = mxGetScalar(Arr_in[2]);
    const float x_s = (float)*mxGetPr(Arr_in[3]);
    const float y_s = (float)*mxGetPr(Arr_in[4]);
           
    mxInitGPU();

    if (mxGetClassID(Arr_in[0]) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    mxGPUArray *Edges = mxGPUCopyFromMxArray(Arr_in[0]);
    const float4 *d_Edges = (float4 *) mxGPUGetData(Edges);
	ISource* d_Sources;
    ISource* host_Sources = new ISource[MaxSources];
            
    cudaMalloc((void**)&d_Sources, sizeof(ISource)*MaxSources);

    kernelIS<<<1,1>>>(d_Edges, d_Sources,/*ISource* Sources,*/ N, MaxSources, x_s, y_s);
    cudaDeviceSynchronize();

    cudaMemcpy( host_Sources, d_Sources, sizeof(ISource)*MaxSources,cudaMemcpyDeviceToHost);

    cudaFree(d_Sources);

    /* Wrap the result up as a MATLAB gpuArray for return. */
    mwSize dims[2] = {1, MaxSources};
    const char* field_names[] = {"x","y"};

    mxArray* OutArray = mxCreateDoubleMatrix(9, MaxSources, mxREAL);
    double * data = (double *) mxGetData(OutArray);
    for (int i = 0; i < MaxSources; ++i) {
        data[i*9] = (double)host_Sources[i].pos.x; 
        data[i*9+1] = (double)host_Sources[i].pos.y;
        data[i*9+2] = (double)host_Sources[i].window.x; 
        data[i*9+3] = (double)host_Sources[i].window.y;
        data[i*9+4] = (double)host_Sources[i].window.z; 
        data[i*9+5] = (double)host_Sources[i].window.w;
        data[i*9+6] = (double)host_Sources[i].edgeid;
        data[i*9+7] = (double)host_Sources[i].parent_source;
        data[i*9+8] = (double)host_Sources[i].reflections;
    }


    cudaFreeHost( host_Sources );

    Arr_out[0] = OutArray;
}