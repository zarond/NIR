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
             887 673 217 673]; % описание ребер комнаты

EdgesRoom(:,[2,4]) = 1000 - EdgesRoom(:,[2,4]);         
EdgesRoom = EdgesRoom';
EdgesRoom = EdgesRoom(:)';
EdgesRoom = d_x*EdgesRoom;

x_ir = fix(M*0.85); % положение слушателя
y_ir = fix(N*0.52);

x_s = fix(M*0.5); % положение источника
y_s = fix(N*0.5);

s = 60; % сигма в пикселях
%U = pointSource(U,s,d_x,x_s,y_s,0); % начальное условие - аппроксимация функции дирака в точке(x_s,y_s)

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
T = hz*1; % время симуляции = 1 секунда

U = pointSource(U,s,d_x,x_s,y_s,0); % начальное условие - аппроксимация функции дирака в точке(x_s,y_s)
U_c = gpuArray(permute(U,[3,2,1])); % загрузка массива на GPU

ImpulseResponse = zeros(1,T,'single');
f1 = figure;
figure(f1);

b=1.0;

t1 = cputime;

for t=0:T-1
    [U_c] = feval(k,U_c,N,M,t+1,v,d_x,timestep,b);
    if (mod(t,100)==0) % отрисовка результата каждые 100 кадров
        U = permute(gather(U_c),[3,2,1]);
        
        %image(U(:,:,mod(t+2,3)+1),'CDataMapping','scaled');
        %colorbar
        
        image(U(:,:,[mod(t+2,3)+1,mod(t+2,3)+1,4])*10+0.5);
        
        drawnow 
    end
    t
    ImpulseResponse(t+1) = U_c(mod(t+2,3)+1,x_ir+1,y_ir+1);   
end

t2 = cputime;
t2-t1 % затраченное общее время на cpu

f2 = figure;
figure(f2);
plot(ImpulseResponse);

%% MexCUDA multiple frames at a time
T = hz*1;%1000;
%U_c = gpuArray(permute(U,[3,2,1])); % Input gpuArray.
ImpulseResponse = zeros(1,T,'single');
ImpulseResponse_c = gpuArray(ImpulseResponse);

mode = 1;
% mode = 0 - считает несколько кадров в течении одного исполнения CUDA ядра. 
% При работе дольше одной секунды CUDA выдает ошибку из-за свойств
% CUDA_KERNEL_TIMEOUT, не рекомендуется
% mode = 1 - считает U и записывает значения IR
% mode = 2 - считает U, использует входной сигнал F и записывает значения IR

if (mode == 0 || mode==1) U = pointSource(U,s,d_x,x_s,y_s,0); end% начальное условие - аппроксимация функции дирака в точке(x_s,y_s)
U_c = gpuArray(permute(U,[3,2,1])); % Input gpuArray.

b = 1;%0.5;

Source = sweeptone(2,2,44100); % Входной сигнал F. Функция из AudioToolkit
%plot(Source)
%title('sinsweep')
Source = cast(Source,'single');

t1 = cputime;
if (mode == 0)
    [U_c, ImpulseResponse_c] = kernel(U_c,ImpulseResponse_c,N,M,0,T,v,d_x,timestep,x_ir,y_ir,mode,b);
end
if (mode == 1)
    [U_c, ImpulseResponse_c] = kernel(U_c,ImpulseResponse_c,N,M,0,T,v,d_x,timestep,x_ir,y_ir,mode,b);
end
if (mode == 2)
    [U_c, ImpulseResponse_c] = kernel(U_c,ImpulseResponse_c,N,M,0,T,v,d_x,timestep,x_ir,y_ir,2,b,x_s,y_s,Source);
end
U = permute(gather(U_c),[3,2,1]);
ImpulseResponse = gather(ImpulseResponse_c);
t2 = cputime; % затраченное общее время на cpu
t2-t1

f1 = figure;
figure(f1);
image(U(:,:,mod(T,3)+2),'CDataMapping','scaled');
colorbar
drawnow   

f2 = figure;
figure(f2);
plot(ImpulseResponse);

if (mode == 2)
    irEstimate = impzest(Source(:),ImpulseResponse(:)); % деконволюция выходного сингала, если был выбран источник
    plot(ImpulseResponse);
end
%%
%Geometry Ray Tracing
D = gpuDevice;
threads = D.MaxThreadsPerBlock;
grid = D.MaxGridSize;

% 1. Create CUDAKernel object.
k = parallel.gpu.CUDAKernel('kernel_Geom.ptx','kernel_Geom.cu','kernelRayTracing'); % ray tracing

SampleCount = N*M*20; % количество лучей
% 2. Set object properties.
k.GridSize = [ceil(SampleCount/threads) 1];
k.ThreadBlockSize = [threads 1];

% 3. Call feval with defined inputs.
T = hz*1;
%Edges = 5*[0,0,5,0,5,0,5,5,5,5,0,5,0,5,0,0]; % комната куб
%Edges = 5*[0,0,6,0,6,0,5,5,5,5,0,5,0,5,0,0]; % комната куб с косым углом
%Edges = 5*[1,0,4,0,5,1,5,4,4,5,1,5,0,4,0,1]; % открытая комната
Edges = EdgesRoom; % комната из room.png
Edges = cast(Edges,'single');
Edges_c = gpuArray(Edges);
ImpulseResponse = zeros(1,T,'single');
ImpulseResponse_c = gpuArray(ImpulseResponse);
f1 = figure;
figure(f1);

maxReflections = 40; % максимальное количество отражений
s_r=0.1; % радиус источника
x_ir=15.3116;
y_ir=7.8059;
x_s=9.0068;
y_s=7.5057;
%y_s=Y_size-7.5057;

t1 = cputime;
[ImpulseResponse_c]=feval(k,Edges_c, size(Edges,2)/4, ImpulseResponse_c, SampleCount,maxReflections,T,v,timestep,x_ir,y_ir,x_s,y_s, s_r);
ImpulseResponse = gather(ImpulseResponse_c);
t2 = cputime; % затраченное общее время на cpu
t2-t1

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
Edges = EdgesRoom;
Edges = cast(Edges,'single');
%MaxSources = 50000; % максимальное число мнимых источников
MaxSources = 64; 

x_ir=15.3116;
y_ir=7.8059;
x_s=9.0068;
%y_s=7.5057;
y_s=Y_size-7.5057;

k = parallel.gpu.CUDAKernel('kernel_visibility.ptx','kernel_visibility.cu','kernelVisibility'); % visibility tracing
k.GridSize = [ceil(MaxSources/threads) 1];
k.ThreadBlockSize = [threads 1];

Visibility = zeros(1,MaxSources,'single');
Visibility_c = gpuArray(Visibility);
ImpulseResponse = zeros(1,T,'single');
ImpulseResponse_c = gpuArray(ImpulseResponse);
Edges_c = gpuArray(Edges);

t1 = cputime;
a = kernel_IS(Edges,size(Edges,2)/4,MaxSources,x_s,y_s); % вычисление мнимых источников
sources_c = gpuArray(cast(a(1:9,:),'single')); 
[Visibility_c,ImpulseResponse_c] = feval(k,Edges_c, size(Edges,2)/4, sources_c, Visibility_c, ImpulseResponse_c, MaxSources,T,v,timestep,x_ir,y_ir); % рассчет видимости из положения приемника
ImpulseResponse = gather(ImpulseResponse_c);
Visibility = gather(Visibility_c);
t2 = cputime; % затраченное общее время на cpu
t2-t1

f1 = figure;
figure(f1);

plot(ImpulseResponse);

%f1 = figure;
%figure(f1);
%disp('f');

Edges_pl = [Edges(1:4:end);Edges(2:4:end);Edges(3:4:end);Edges(4:4:end)];

f2 = figure;
figure(f2);

clf
hold on
plot([Edges_pl(1,:);Edges_pl(3,:)],[Edges_pl(2,:);Edges_pl(4,:)],'Color','k');
%plot(a(1,:),a(2,:),'*');
%a=a+rand(size(a));
%plot([a(3,:);a(1,:);a(5,:)],[a(4,:);a(2,:);a(6,:)],'--','Color','g')
%plot([a(3,:);a(1,:);a(5,:)],[a(4,:);a(2,:);a(6,:)],'--','Color','r')
a_vis = a(:,Visibility>0);
a_invis = a(:,Visibility<0);
plot(a_vis(1,:),a_vis(2,:),'*','Color','g');
plot(a_invis(1,:),a_invis(2,:),'*','Color','r');
plot([a_vis(3,:);a_vis(1,:);a_vis(5,:)],[a_vis(4,:);a_vis(2,:);a_vis(6,:)],'-','Color','g')
plot([a_invis(3,:);a_invis(1,:);a_invis(5,:)],[a_invis(4,:);a_invis(2,:);a_invis(6,:)],'-','Color','r')
plot(x_ir,y_ir,'o')
hold off
% drawnow
%end
