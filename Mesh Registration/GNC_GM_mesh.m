function [ang_err,tran_error] = GNC_GM_mesh(correspondences,std,gtT)
d=1;

w=ones(1,size(correspondences,2));
[R,t,dstar,times] = method_RCQP( correspondences, 'header_all' ,w);


eps=chi2inv(0.99, 3);

residues=r(correspondences,std,R,t);
tau=max(residues);
mu_GM=2*tau/eps;
ti=0;
r_m=sum(residues);


while(mu_GM>1)
    d;
    ti=ti+1;

    w=gncWeightsUpdategm(mu_GM, residues, eps,w);
    sum(w);
    r_p=sum(residues.*w);
    
    [R,t,dstar,times] = method_RCQP( correspondences, 'header_all' ,w);
    mu_GM=mu_GM/1.4;
    residues=r(correspondences,std,R,t);
    
    
    d= abs([r_p-r_m]/r_m);
    r_m=r_p;
end



% 
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
 

function [weights] = gncWeightsUpdategm(mu, residuals, epsl,weights)
for k_k = 1:length(residuals)   
        weights(k_k) = ( (epsl*mu)/(residuals(k_k)+epsl*mu) )^2;

end
end


