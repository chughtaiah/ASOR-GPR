function [ang_err,tran_error] = ESOR_mesh(correspondences,std,gtT)
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
    w=sorWeightsUpdate(residues,w);
    r_p=sum(residues.*w);   

    if (sum(w)==0 | t>max_iterations)
        break
    end
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




function [weights] = sorWeightsUpdate(residuals,weights)
% 

eps=chi2inv(0.99, 3);
ravg=sum(residuals.*weights)/sum(weights);
ravg=max(ravg,eps);


for k = 1:length(residuals)    

weights(k)=1/(1+exp(.5*(residuals(k)-ravg) ) ); 


end
% end

end


