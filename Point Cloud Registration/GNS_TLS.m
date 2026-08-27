function [ang_err,tran_error]=GNS_TLS(lpts_3d,lpts_3d_,ln_ele,lR_gt,lt_gt,std)
result=horns(lpts_3d',lpts_3d_','weights',ones(ln_ele,1));
w=ones(1,ln_ele);
d=1;
eps=chi2inv(0.99, 3);
% eps=10;
r_empty=r(lpts_3d',lpts_3d_',result.R,result.t,std);
r_m=sum(r_empty);
tau=max(r_empty);
mu_GNS=eps;
mu_GNS=mu_GNS/(2*tau-eps);
t=0;
residues=r_empty;

while(d>10^-5)
    t=t+1;

    w=gncWeightsUpdate(mu_GNS, residues, eps,w);
    r_p=sum(residues.*w);
    result=horns(lpts_3d',lpts_3d_','weights',w);
    mu_GNS=1.4*mu_GNS;
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

% function flag = areBinaryWeights(weights)
% % Check the input vector is (approximately) a binary vectory
% flag = true;
% th = 1e-12;
% for i = length(weights):-1:1
%     if weights(i) > th && abs(1 - weights(i)) > th
%         flag = false;
%         break
%     end
% end
% end

function [weights] = gncWeightsUpdate(mu, residuals, epsl,weights)
th1 = (mu+1)/mu * epsl;
th2 = (mu)/(mu+1) * epsl; % th1 > th2
for k_k = 1:length(residuals)
    if residuals(k_k) - th1 >= 0
        weights(k_k) = 0;
    elseif residuals(k_k) - th2 <= 0
        weights(k_k) = 1;
    else
        weights(k_k) = sqrt( epsl*mu*(mu+1)/residuals(k_k) ) - mu;
%         assert(weights(k)>= 0 && weights(k) <=1, 'weights %g calculation wrong!', weights(k));
    end
end
end
