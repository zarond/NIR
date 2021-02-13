//nvcc -ptx "E:\семестр 7\НИР\kernel_visibility.cu" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64"
//nvcc -ptx "E:\семестр 7\НИР\kernel_visibility.cu" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64" -gencode arch=compute_35,code=sm_35 -rdc=true
#include "inc\helper_math.h"
        
#define eps 0.000001
#define pi 3.141592654
        
inline __device__ float cross(float2 a, float2 b){return a.x*b.y-a.y*b.x;}
inline __device__ float2 xy(float4 a) {return make_float2(a.x,a.y);}
inline __device__ float2 zw(float4 a) {return make_float2(a.z,a.w);}
inline __device__ bool operator==(float4 a, float b) {return (a.x==b && a.y==b && a.z==b && a.w==b);}
inline __device__ bool operator<=(float2 a, float b) {return (a.x<=b && a.y<=b);}
inline __device__ float4 make_float4(float2 a, float2 b) {return make_float4(a.x,a.y,b.x,b.y);}      
inline __device__ float2 normal(float2 a){return make_float2(-a.y,a.x);}

struct ISource{
    float2 pos;
    float timeoffset;
    float4 window;
    int edgeid;
    int parent_source;
    int reflections;
    int difractions;
};

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

inline __device__ bool ComputeVisibility(float4 * Edges, const unsigned int N, float2 pos, float2 direction, const ISource source, float &r){
    int chosenEdgeid;
    float minR=1.0e+37;

    float2 twindow = IntersectRayEdge(pos, direction,source.window);
    if (!(twindow.x > 0.0 && (twindow.y >= 0.0 && twindow.y <= 1.0)) && source.reflections != 0){ // не пересекается с окном источника
        r=-1;
        return false;
    }
    
    for (int i=0;i<N;++i){
        float4 edge = Edges[i];
        float2 t = IntersectRayEdge(pos, direction, edge);
        if (t.x > 0.0 && (t.y >= 0.0 && t.y <= 1.0) && (t.x<=minR))
        {
            chosenEdgeid = i;
            minR=t.x;
        }
    }
    float tmp = sqrt(dot(source.pos - pos,source.pos - pos));
    if (source.reflections == 0 && minR > tmp){
        r = tmp;
        return true;
    }
    if (minR<1.0e+35 && chosenEdgeid == source.edgeid){
        r = minR;
        return true;
    }
    r=-1.0;
    return false;
}        
 
__global__ void kernelVisibility(const float *Edges_f, const unsigned int N, const float *Sources, float* visibility, float *IR, const unsigned int M,const int T,const float v,const float d_t,const float x_ir,const float y_ir)
{ // visibility возвращает расстояние по лучу от слушателя к ребру, относительно которого построен источник
    float4 *Edges = (float4*)Edges_f;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= M) return;
    ISource source = {make_float2(Sources[i*9],Sources[i*9+1]), 0,
    make_float4(Sources[i*9+2],Sources[i*9+3],Sources[i*9+4],Sources[i*9+5]),int(round(Sources[i*9+6])),int(round(Sources[i*9+7])),int(round(Sources[i*9+8])),0};

    float2 pos = make_float2(x_ir,y_ir);
    float2 direction = normalize(source.pos - pos);

    float r=-1;
    bool sign = (source.reflections % 2 == 0);

    bool hitsource = ComputeVisibility(Edges, N, pos, direction, source,r);
    visibility[i] = r;

    if (hitsource==false) return;

    r = sqrt(dot(source.pos - pos,source.pos - pos));
    int ind = int(r/(v*d_t));
    if (ind<T && ind>=0) 
        r = (sign)? r : -r;
        atomicAdd(&IR[ind],2*pi/r);

    //i = blockIdx.x * blockDim.x + threadIdx.x;
    //if (i<T) IR[i]=r;
}