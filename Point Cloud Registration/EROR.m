function [ang_err,tran_error]=EROR(lpts_3d,lpts_3d_,ln_ele,lR_gt,lt_gt,std)
result=horns(lpts_3d',lpts_3d_','weights',ones(ln_ele,1));
w=ones(1,ln_ele);
d=1;
t=0;
residues=r(lpts_3d',lpts_3d_',result.R,result.t,std);
r_m=sum(residues);


while(d>10^-5)

    t=t+1;

    w=erorWeightsUpdate(residues, w);

    if sum(w)<=0.05*numel(w)
        break
    end
    result=horns(lpts_3d',lpts_3d_','weights',w);
    residues=r(lpts_3d',lpts_3d_',result.R,result.t,std);
  
    r_p=sum(residues.*w);

    d=norm(r_p-r_m)/norm(r_m);
    r_m=r_p;
end
ang_err=AngularError(result.R,lR_gt)*180/pi;
tran_error=norm(result.t-lt_gt');
end

function [residues]=r(a,P,RR,tt,noise)
    
    residues=((RR*a+tt)-P).^2;
    residues=sum(residues,1);
    residues=residues/(noise^2);

end



function [weights] = erorWeightsUpdate(residuals,weights)
rmax=max(residuals.*weights);
rmin=min(residuals.*weights);
eps=chi2inv(0.99, 3);
v=(rmax+rmin)/2;
v=max(v,eps);

for k = 1:length(residuals)
weights(k)=1/(residuals(k)/v+1);
end


end

