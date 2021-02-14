function [out] = pointSource(U,s,d_x,x_s,y_s,mode)
% апроксимация функции Дирака с помощью нормального распределения (mode=0),
% одной точки (mode = 1), и синуса (mode = 2)
sigma = s*d_x;
for i=y_s-2*s:y_s+2*s
    for j=x_s-2*s:x_s+2*s
        inp = 0;
        r=(i-y_s)^2+(j-x_s)^2;
        if (mode==0)
            if (r<=(3*s)^2) inp=normpdf(sqrt(r),0,sigma); end
        end
        if (mode==1)
            if (r<=0) inp=1/(d_x^2); end
        end
        if (mode==2)
            if (r<=(3*s)^2 && r~=0) inp=sin(5*pi*sqrt(r))/(sqrt(r)*pi); else if (r==0) inp = 5; end; end
        end
        %U(i+1,j+1,[1])=inp; % импульс
        U(i+1,j+1,[2])=inp; % импульс
    end
end
out = U;
end

