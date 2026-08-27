function [ang_err,tran_error]=ASOR(lpts_3d,lpts_3d_,ln_ele,lR_gt,lt_gt,std)

result=horns(lpts_3d',lpts_3d_','weights',ones(ln_ele,1));
w=ones(1,ln_ele);
omega=ones(1,ln_ele);
d=1;
t=0;
residues=r(lpts_3d',lpts_3d_',result.R,result.t,std);
r_m=sum(residues.*w);    
b=10000;
a0=.5;
tau=gamma(a0+.5)/gamma(a0);

while(d>10^-5)

    t=t+1;

    omega=gammaOmegaUpdate(residues,b,a0,tau);
    b=fun_b(residues,omega,b,a0);
    w=asorWeightsUpdate(residues,omega,b,a0);
    result=horns(lpts_3d',lpts_3d_','weights',w);
    residues=r(lpts_3d',lpts_3d_',result.R,result.t,std);
    r_p=sum(residues.*w);

    d=norm(r_p-r_m)/norm(r_m);
    r_m=r_p;
end
% end

ang_err=AngularError(result.R,lR_gt)*180/pi;
tran_error=norm(result.t-lt_gt');
end

function [residues]=r(a,P,RR,tt,noise)
    
    residues=((RR*a+tt)-P).^2;
    residues=sum(residues,1);
    residues=residues/(noise^2);

end


function [omegas] = gammaOmegaUpdate(residuals,bin,a0,tau_i)

for k = 1:length(residuals)    

omegas(1,k)=1/(1+1/tau_i*exp(.5*(residuals(k)))*bin^a0/((residuals(k))/2+bin)^(a0+.5) ); 

end
end

function [bl] = fun_b(residuals,omegas,bin,a0)

A=10000;
B=1000;

A_bar=A+a0*sum(1-omegas)-1;
B_bar=(B+(a0+.5)*sum((1-omegas)./(residuals/2+bin)) );
bl=A_bar/B_bar;

end

function [weights] = asorWeightsUpdate(residuals,omegas,bin,a0)

for k = 1:length(residuals)    

weights(1,k)=omegas(1,k)+(1-omegas(1,k))*(a0+.5)/(residuals(1,k)/2+bin)   ; 

end
end


