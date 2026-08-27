function [ang_err,tran_error]=LS(lpts_3d,lpts_3d_,ln_ele,lR_gt,lt_gt,std)
result=horns(lpts_3d',lpts_3d_','weights',ones(ln_ele,1));
% w=ones(1,ln_ele);
% d=1;
% t=0;
% residues=r(lpts_3d',lpts_3d_',result.R,result.t,std);
% r_m=sum(residues);
% 
% % eps=10^-100;
% % eps=.99;
% 
% 
%     w=sorWeightsUpdate(residues, w);
%     r_p=sum(residues.*w);
%     result=horns(lpts_3d',lpts_3d_','weights',w);
%     residues=r(lpts_3d',lpts_3d_',result.R,result.t,std);
% 
%     d=norm(r_p-r_m)/norm(r_m);
%     r_m=r_p;



ang_err=AngularError(result.R,lR_gt)*180/pi;
tran_error=norm(result.t-lt_gt');
end

function [residues]=r(a,P,RR,tt,noise)
    
    residues=((RR*a+tt)-P).^2;
    residues=sum(residues,1);
    residues=residues/(noise^2);

end


function [weights] = sorWeightsUpdate(residuals,weights)


for k = 1:length(residuals)

weights(k)=1/(1+exp(.5*(residuals(k)-10) ) ) ;

end

end