function [ang_err,tran_error] = EROR_mesh(correspondences,std,gtT)
d=1;

w=ones(1,size(correspondences,2));
[R,t,dstar,times] = method_RCQP( correspondences, 'header_all' ,w);

residues=r(correspondences,std,R,t);
r_m=sum(residues);
ti=0;
th=10^-4;
max_iterations=400;

while(d>th)
    d;
    ti=ti+1;

    w=rorWeightsUpdate(residues,w); 
    r_p=sum(residues.*w);

%     if sum(w)<=0.05*numel(w)
    if (sum(w)==0 | ti>max_iterations)
        break
    end
    [R,t,dstar,times] = method_RCQP( correspondences, 'header_all' ,w);
    residues=r(correspondences,std,R,t);
    
%     
    d=abs([r_p-r_m]/r_m);
    r_m=r_p;
end


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
% 


function [weights] = rorWeightsUpdate(residuals,weights)
rmax=max(residuals.*weights);
rmin=min(residuals.*weights);
eps=chi2inv(0.99, 3);
v=(rmax+rmin)/2;
v=max(v,eps);

for k = 1:length(residuals)
weights(k)=1/(residuals(k)/v+1);
end

end
