//nvcc -ptx "E:\семестр 7\НИР\kernel_Geom.cu" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64"
//nvcc -ptx "E:\семестр 7\НИР\kernel_Geom.cu" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64" -gencode arch=compute_52,code=sm_52 -rdc=true
//nvcc -ptx "D:\семестр 7\НИР\kernel_Geom.cu" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.28.29333\bin\Hostx64\x64" -gencode arch=compute_52,code=sm_52 -rdc=true
#include "inc\helper_math.h"
        
#define eps 0.000001f
#define pi 3.141592654f
        
inline __device__ float cross(float2 a, float2 b){return a.x*b.y-a.y*b.x;}

__device__ bool ComputeIntersection(float4 * Edges, const unsigned int N, float2 pos, float2 direction, float2 &newpos, float2 &newdirection, float &r,const float s_r, const float x_s, const float y_s){
    float4 chosenEdge;
    float minR=1.0e+37f;
    bool hitSource = false;
    
    float2 SourceDirection = pos - make_float2(x_s,y_s);
    float b = dot( SourceDirection, direction );
    float c = dot( SourceDirection, SourceDirection ) - s_r*s_r;
    float h = b*b - c;
    if( h>=0.0f && b<=0.0f){ // intersection with source
        h = sqrt(h);
        minR = -b-h;
        hitSource = true;
    }

    for (int i=0;i<N;++i){
        float4 edge = Edges[i];
        SourceDirection = make_float2(pos.x - edge.x,pos.y - edge.y);
        float2 edgedir = make_float2(edge.z-edge.x,edge.w-edge.y);
        float2 v3 = make_float2(-direction.y, direction.x);
        //v3 = normalize(v3);

        float dotv = dot(edgedir,v3);
        //if (abs(dotv) < eps)
        //   continue;

        float t1 = cross(edgedir,SourceDirection) / dotv;
        float t2 = dot(SourceDirection,v3) / dotv;

        if (t1 >= 0.0f && (t2 >= 0.0f && t2 <= 1.0f) && (t1<=minR))
        {
            chosenEdge = edge;
            minR=t1;
            hitSource = false;
        }
    }
    if (minR<1.0e+35f){
        r += minR;//+eps;
        if (hitSource == true) return true;
        float2 edgedir = make_float2(chosenEdge.z-chosenEdge.x,chosenEdge.w-chosenEdge.y);
        float2 n = normalize(make_float2(-edgedir.y,edgedir.x));
        newdirection = direction-2.0f*dot(n,direction)*n;
        newdirection = normalize(newdirection);
        newpos = pos+(1.0f-eps)*minR*direction;// + newdirection*eps;
        return false;
    }
    r=-1.0f;
    return false;
}        
 
__global__ void kernelRayTracing(const float *Edges_f, const unsigned int N, float *IR, const unsigned int SampleCount,const int maxReflections,const int T,const float v,const float d_t,const float x_ir,const float y_ir, const float x_s,const float y_s, const float s_r=0.1f)
{
    float4 *Edges = (float4*)Edges_f;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= SampleCount) return;
	
    float2 pos = make_float2(x_ir,y_ir);
    float phi = 2.0f*float(i)*pi/SampleCount;
    float2 direction = make_float2(cos(phi),sin(phi));

    float r=0.0f;
    float2 newpos;
    float2 newdirection;
    bool sign = true;

    i=0;
    for (;i<maxReflections;++i){
        bool hitsource = ComputeIntersection(Edges, N, pos, direction, newpos, newdirection, r ,s_r, x_s, y_s);
        if (hitsource) break;
        pos = newpos;
        direction = newdirection;
        sign=!sign;
    }
    if (i>=maxReflections) return;
    int ind = int(r/(v*d_t));
    if (ind<T && ind>=0) 
        r = (sign)? r : -r;
        float g = 2.0f*pi/SampleCount;
        g = (sign)? g : -g;
        atomicAdd(&IR[ind],g);

}