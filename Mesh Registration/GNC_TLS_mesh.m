function [ang_err,tran_error] = GNC_TLS_mesh(correspondences,std,gtT)
d=1;

w=ones(1,size(correspondences,2));
[R,t,dstar,times] = method_RCQP( correspondences, 'header_all' ,w);

% T = Pose(t,R);

% norm(gt_T.t-T.t)
% norm(gt_T.R-T.R)*180/pi

eps=chi2inv(0.99, 3);
% eps=10;
r_empty=r(correspondences,std,R,t);
% size(r_empty)
% plot(r_empty)
r_m=sum(r_empty);
tau=max(r_empty);
mu_GNS=eps;
mu_GNS=mu_GNS/(2*tau-eps);
ti=0;
residues=r_empty;
th=10^-4;
max_iterations=400;

while(d>th)
    d;
    ti=ti+1;
    
    if (ti>max_iterations)
        break
    end
    
    w=gncWeightsUpdate(mu_GNS, residues, eps,w);
%     sum(w)
    r_p=sum(residues.*w);
    
    [R,t,dstar,times] = method_RCQP( correspondences, 'header_all' ,w);
    mu_GNS=1.4*mu_GNS;
    residues=r(correspondences,std,R,t);
    
    
    d= abs([r_p-r_m]/r_m);
    r_m=r_p;
    
%     if areBinaryWeights(w)
%         break
%     end
end

% ang_err=AngularError(result.R,lR_gt)*180/pi;
% % result.R-lR_gt;
% tran_error=norm(result.t-lt_gt');
% % end
% 
T = Pose(t,R);
ang_err=AngularError(gtT.R,T.R)*180/pi;
tran_error=norm(gtT.t-T.t);
% norm(gt_T.R-T.R)*180/pi
end


function [residues]=r(corr,noise,RR,tt)
    
    residues=zeros(1,numel(corr));

for i=1:numel(corr)
  
  residues(i)=distSq2(corr(i).model,corr(i).point,RR,tt);
    
end
residues=residues/noise^2;
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


