function [ang_err,tran_error]=GNS_GM(lpts_3d,lpts_3d_,ln_ele,lR_gt,lt_gt,std)
result=horns(lpts_3d',lpts_3d_','weights',ones(ln_ele,1));
w=ones(1,ln_ele);
d=1;
eps=chi2inv(0.99, 3);
% eps=10;
residues=r(lpts_3d',lpts_3d_',result.R,result.t,std);
tau=max(residues);
mu_GM=2*tau/eps;
r_m=sum(residues);

t=0;

% while(mu_GM>1)
while(mu_GM>1)
    t=t+1;

    w=gncWeightsUpdategm(mu_GM, residues, eps,w);
    r_p=sum(residues.*w);
    result=horns(lpts_3d',lpts_3d_','weights',w);
    mu_GM=mu_GM/1.4;
    residues=r(lpts_3d',lpts_3d_',result.R,result.t,std);
    
    
    d=norm([r_p-r_m])/norm(r_m);
    r_m=r_p;
end
ang_err=AngularError(result.R,lR_gt)*180/pi;
% result.R-lR_gt;
tran_error=norm(result.t-lt_gt');
end

function [residues]=r(a,P,RR,tt,noise)
    
    residues=((RR*a+tt)-P).^2;
    residues=sum(residues,1);
    residues=residues/(noise^2);

end

function flag = areBinaryWeights(weights)
% Check the input vector is (approximately) a binary vectory
flag = true;
th = 1e-12;
for i = length(weights):-1:1
    if weights(i) > th && abs(1 - weights(i)) > th
        flag = false;
        break
    end
end
end

% 
function [weights] = gncWeightsUpdategm(mu, residuals, epsl,weights)
for k_k = 1:length(residuals)   
        weights(k_k) = ( (epsl*mu)/(residuals(k_k)+epsl*mu) )^2;

end
end

