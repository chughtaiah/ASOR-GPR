function [ang_err,tran_error] = ASOR_mesh(correspondences,std,gtT)
d=1;
w=ones(1,size(correspondences,2));

[R,t,dstar,times] = method_RCQP( correspondences, 'header_all' ,w);

residues=r(correspondences,std,R,t);
r_m=sum(residues);
ti=0;
b=10000;
a0=.5;
th=10^-4;
max_iterations=400;

while(d>th)
    d;
    ti=ti+1;
    if (ti>max_iterations)
        break
    end
    omega=gammaOmegaUpdate(residues,b,a0);
    b=fun_b(residues,omega,b,a0);
    w=gammaWeightsUpdate(residues,omega,b,a0);
    r_p=sum(residues.*w);   


    [R,t,dstar,times] = method_RCQP( correspondences, 'header_all' ,w);
    residues=r(correspondences,std,R,t);
       
    d=abs([r_p-r_m]/r_m);
    r_m=r_p;
end



T = Pose(t,R);
ang_err=AngularError(gtT.R,T.R)*180/pi;
tran_error=norm(gtT.t-T.t);
end


function [residues]=r(corr,noise,RR,tt)
    
    residues=zeros(1,numel(corr));

for i=1:numel(corr)
  
  residues(i)=distSq2(corr(i).model,corr(i).point,RR,tt);
    
end
residues=residues/noise^2;
end
% 

function [omegas] = gammaOmegaUpdate(residuals,bin,a0)

for k = 1:length(residuals)    

omegas(1,k)=1/(1+1/sqrt(pi)*exp(.5*(residuals(k)))*bin^a0/((residuals(k))/2+bin)^(a0+.5) ); 

end
end

function [bl] = fun_b(residuals,omegas,bin,a0)

A=10000;
B=1000;

bl=A+a0*sum(1-omegas);
bl=bl/(B+(a0+.5)*sum((1-omegas)./(residuals/2+bin)) );

end

function [weights] = gammaWeightsUpdate(residuals,omegas,bin,a0)

for k = 1:length(residuals)    

weights(1,k)=omegas(1,k)+(1-omegas(1,k))*(a0+.5)/(residuals(1,k)/2+bin)   ; 

end
end


