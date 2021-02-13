hz = 44100; % звук 44100 герц, 16-битный
timestep = 1.0/hz; % ход симуляции для 44100 герц
v = 331; % скорость звука м/с
d_x = v*timestep; %  шаг сетки в пространстве 0.0075 м = 7.5 мм, но для симуляции надо брать больше X2
d_x = d_x*2;
X_size = 25; % размеры комнаты
Y_size = 25;
N = fix(Y_size / d_x); % размеры в пикселях
M = fix(X_size / d_x);

U=[];
try
    U = imread('room.png'); % прочитать файл с геометрией комнаты
catch
end
if (isempty(U)) % если не прочитался файл - пустая комната
    U = zeros(N,M,4,'single');
    %U(:,:,4)=1.0;
    U(2:end-1,2:end-1,4)=1.0;
else % если прочитался - получить параметры
    U = cast(U,'single');
    U = U./255;
    N = size(U,1);
    M = size(U,2);
    X_size = M*d_x; % размеры комнаты
    Y_size = N*d_x;
    U(:,:,4)=1.0-U(:,:,1);%*0.999;
    U(:,:,[1,2,3])=0;
end
EdgesRoom = [3 3 3 994
             3 994 1195 994
             1195 994 1195 3
             1195 3 3 3
             214 673 214 175
             214 175 887 175
             887 175 887 289
             887 289 883 289
             883 289 883 179
             883 179 218 179
             218 179 218 669
             218 669 883 669
             883 669 883 618
             883 618 887 618
             887 618 887 673
             887 673 217 673];
EdgesRoom = EdgesRoom';
EdgesRoom = EdgesRoom(:)';
EdgesRoom = d_x*EdgesRoom;

x_ir = fix(M*0.85); % положение слушателя
y_ir = fix(N*0.52);

x_s = fix(M*0.5); % положение источника
y_s = fix(N*0.5);

s = 60; % сигма в пикселях
U = pointSource(U,s,d_x,x_s,y_s,0); % начальное условие - аппроксимация функции дирака в точке(x_s,y_s)

image(U(:,:,[2,2,4]));%,'CDataMapping','scaled');

%% CUDA parallel computing toolbox разностная схема
D = gpuDevice;
threads = D.MaxThreadsPerBlock;
grid = D.MaxGridSize;

% 1. Create CUDAKernel object.
k = parallel.gpu.CUDAKernel('kernel_simple.ptx','kernel_simple.cu','kernel'); % обычная итерация
k1 = parallel.gpu.CUDAKernel('kernel_simple.ptx','kernel_simple.cu','kernelAndSetIR'); % записывает значения в точке (x_ir,y_r)
k2 = parallel.gpu.CUDAKernel('kernel_simple.ptx','kernel_simple.cu','kernelAndSetIRAndSource'); % плюс возмущения в источнике

% 2. Set object properties.
k.GridSize = [ceil(N*M/threads) 1];
k.ThreadBlockSize = [threads 1];

% 3. Call feval with defined inputs.
T = hz*1;
U_c = gpuArray(permute(U,[3,2,1])); % Input gpuArray.

ImpulseResponse = zeros(1,T,'single');
f1 = figure;
figure(f1);

b=1.0;

t1 = cputime;

for t=0:T-1
    [U_c] = feval(k,U_c,N,M,t+1,v,d_x,timestep,b);
    if (mod(t,100)==0)
        U = permute(gather(U_c),[3,2,1]);
         image(U(:,:,mod(t+2,3)+1),'CDataMapping','scaled');
        %image(U(:,:,[mod(t+2,3)+1,mod(t+2,3)+1,4])*10+0.5);
        colorbar
        drawnow 
    end
    %t
    ImpulseResponse(t+1) = U_c(mod(t+2,3)+1,x_ir+1,y_ir+1);   
end

t2 = cputime;
t2-t1

f2 = figure;
figure(f2);
plot(ImpulseResponse);

%% MexCUDA multiple frames at a time
T = hz*1;%1000;
U_c = gpuArray(permute(U,[3,2,1])); % Input gpuArray.
ImpulseResponse = zeros(1,T,'single');
ImpulseResponse_c = gpuArray(ImpulseResponse);

mode = 1;
b = 2;%0.5;

%Source = sin(linspace(0,50,T));
Source = sweeptone(2,2,44100);
%plot(Source)
%title('sinsweep')
Source = cast(Source,'single');
%Source = zeros(1,T,'single'); Source(1)=1.0;

t1 = cputime;
[U_c, ImpulseResponse_c] = kernel(U_c,ImpulseResponse_c,N,M,0,T,v,d_x,timestep,x_ir,y_ir,mode,b);
%[U_c, ImpulseResponse_c] = kernel(U_c,ImpulseResponse_c,N,M,0,T,v,d_x,timestep,x_ir,y_ir,2,b,x_s,y_s,Source);

U = permute(gather(U_c),[3,2,1]);
ImpulseResponse = gather(ImpulseResponse_c);
t2 = cputime;
t2-t1

f1 = figure;
figure(f1);
image(U(:,:,mod(T,3)+2),'CDataMapping','scaled');

        colorbar
drawnow   

f2 = figure;
figure(f2);
plot(ImpulseResponse);

irEstimate = impzest(Source(:),ImpulseResponse(:));
%plot(irEstimate);
%%
%Geometry Ray Tracing
D = gpuDevice;
threads = D.MaxThreadsPerBlock;
grid = D.MaxGridSize;

% 1. Create CUDAKernel object.
k = parallel.gpu.CUDAKernel('kernel_Geom.ptx','kernel_Geom.cu','kernelRayTracing'); % ray tracing

SampleCount = N*M*20;
% 2. Set object properties.
k.GridSize = [ceil(SampleCount/threads) 1];
k.ThreadBlockSize = [threads 1];

% 3. Call feval with defined inputs.
T = hz*1;
%Edges = 5*[0,0,5,0,5,0,5,5,5,5,0,5,0,5,0,0];
%Edges = 5*[0,0,6,0,6,0,5,5,5,5,0,5,0,5,0,0]
Edges = 5*[1,0,4,0,5,1,5,4,4,5,1,5,0,4,0,1];
Edges = EdgesRoom;
Edges = cast(Edges,'single');
Edges_c = gpuArray(Edges);
ImpulseResponse = zeros(1,T,'single');
ImpulseResponse_c = gpuArray(ImpulseResponse);
f1 = figure;
figure(f1);

maxReflections = 40;
s_r=0.1;
x_ir=15.3116;
y_ir=7.8059;
x_s=9.0068;
y_s=7.5057;

[ImpulseResponse_c]=feval(k,Edges_c, size(Edges,2)/4, ImpulseResponse_c, SampleCount,maxReflections,T,v,timestep,x_ir,y_ir,x_s,y_s, s_r);

ImpulseResponse = gather(ImpulseResponse_c);
plot(ImpulseResponse);

%%
% Image Source

D = gpuDevice;
threads = D.MaxThreadsPerBlock;
grid = D.MaxGridSize;
T = hz*1;

Edges = 5*[0,0,5,0,5,0,5,5,5,5,0,5,0,5,0,0];
%Edges = 5*[0,0,6,0,6,0,5,5,5,5,0,5,0,5,0,0];
%Edges = 5*[1,0,4,0,5,1,5,4,4,5,1,5,0,4,0,1];
%Edges = EdgesRoom;
Edges = cast(Edges,'single');
MaxSources = 64;

x_ir=15.3116;
y_ir=7.8059;
x_s=9.0068;
y_s=7.5057;

k = parallel.gpu.CUDAKernel('kernel_visibility.ptx','kernel_visibility.cu','kernelVisibility'); % visibility tracing
k.GridSize = [ceil(MaxSources/threads) 1];
k.ThreadBlockSize = [threads 1];

Visibility = zeros(1,MaxSources,'single');
Visibility_c = gpuArray(Visibility);
ImpulseResponse = zeros(1,T,'single');
ImpulseResponse_c = gpuArray(ImpulseResponse);
Edges_c = gpuArray(Edges);

tic
a = kernel_IS(Edges,size(Edges,2)/4,MaxSources,x_s,y_s);
sources_c = gpuArray(cast(a(1:9,:),'single'));
[Visibility_c,ImpulseResponse_c] = feval(k,Edges_c, size(Edges,2)/4, sources_c, Visibility_c, ImpulseResponse_c, MaxSources,T,v,timestep,x_ir,y_ir);
toc

f1 = figure;
figure(f1);

ImpulseResponse = gather(ImpulseResponse_c);
Visibility = gather(Visibility_c);
plot(ImpulseResponse);

%f1 = figure;
%figure(f1);
disp('f');

%Edges_pl = [Edges(1:2:end);Edges(2:2:end)];
Edges_pl = [Edges(1:4:end);Edges(2:4:end);Edges(3:4:end);Edges(4:4:end)];

f2 = figure;
figure(f2);

%for t=0:1000
% x_s=12.5+10*sin(t/200);
% y_s=12.5+10*cos(t/200);
clf
hold on
% tic
% a = kernel_IS(Edges,4,MaxSources,x_s,y_s);
% toc
%plot(Edges_pl(1,:),Edges_pl(2,:),'Color','k');
plot([Edges_pl(1,:);Edges_pl(3,:)],[Edges_pl(2,:);Edges_pl(4,:)],'Color','k');
%plot(a(1,:),a(2,:),'*');
%a=a+rand(size(a));
%plot([a(3,:);a(1,:);a(5,:)],[a(4,:);a(2,:);a(6,:)],'--','Color','g')
%plot([a(3,:);a(1,:);a(5,:)],[a(4,:);a(2,:);a(6,:)],'--','Color','r')
a_vis = a(:,Visibility>0);
a_invis = a(:,Visibility<0);
%a_vis = a_vis(:,end-10:end);
plot(a_vis(1,:),a_vis(2,:),'*','Color','g');
plot(a_invis(1,:),a_invis(2,:),'*','Color','r');
plot([a_vis(3,:);a_vis(1,:);a_vis(5,:)],[a_vis(4,:);a_vis(2,:);a_vis(6,:)],'-','Color','g')
plot([a_invis(3,:);a_invis(1,:);a_invis(5,:)],[a_invis(4,:);a_invis(2,:);a_invis(6,:)],'-','Color','r')
plot(x_ir,y_ir,'o')
hold off
% drawnow
%end



%%
%Ray Marching
T = hz*1;
ImpulseResponse = zeros(1,T,'single');
ImpulseResponse_c = gpuArray(ImpulseResponse);
SampleCount = T;%N*M;
maxSteps = 2*2048;
s_r=0.1/d_x;
x_ir = fix(M*0.85); % положение слушателя
y_ir = fix(N*0.52);

x_s = fix(M*0.5); % положение источника
y_s = fix(N*0.5);

SF = createSF(0,N,M);

f2 = figure;
figure(f2);

ImpulseResponse_c = kernel_RM(SF,N,M,ImpulseResponse_c,SampleCount,maxSteps,T,v,timestep,x_ir,y_ir,x_s,y_s, d_x, s_r);
ImpulseResponse = gather(ImpulseResponse_c);
plot(ImpulseResponse);


%%
%Partial Derivative Equation;
c = 1;
a = 0;
f = 0;
m = 1;

numberOfPDE = 1;
model = createpde(numberOfPDE);
geometryFromEdges(model,@squareg);
pdegplot(model,'EdgeLabels','on'); 
ylim([-1.1 1.1]);
axis equal
title 'Geometry With Edge Labels Displayed';
xlabel x
ylabel y

specifyCoefficients(model,'m',m,'d',0,'c',c,'a',a,'f',f);

applyBoundaryCondition(model,'dirichlet','Edge',[2,4],'u',0);
applyBoundaryCondition(model,'neumann','Edge',([1 3]),'g',0);

generateMesh(model,'Hmax',d_x);
figure
pdemesh(model);
ylim([-1.1 1.1]);
axis equal
xlabel x
ylabel y

u0 = @(location) atan(cos(pi/2*location.x));
ut0 = @(location) 3*sin(pi*location.x).*exp(sin(pi/2*location.y));
setInitialConditions(model,u0,ut0);

n = 31;
tlist = linspace(0,5,n);

model.SolverOptions.ReportStatistics ='on';
result = solvepde(model,tlist);

u = result.NodalSolution;

figure
umax = max(max(u));
umin = min(min(u));
for i = 1:n
    pdeplot(model,'XYData',u(:,i),'ZData',u(:,i),'ZStyle','continuous',...
                  'Mesh','off','XYGrid','on','ColorBar','off');
    axis([-1 1 -1 1 umin umax]); 
    caxis([umin umax]);
    xlabel x
    ylabel y
    zlabel u
    M(i) = getframe;
end
