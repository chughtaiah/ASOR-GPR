function [ang_err,tran_error]=ROR(lpts_3d,lpts_3d_,ln_ele,lR_gt,lt_gt,std)
result=horns(lpts_3d',lpts_3d_','weights',ones(ln_ele,1));
w=ones(1,ln_ele);
d=1;
t=0;
residues=r(lpts_3d',lpts_3d_',result.R,result.t,std);
r_m=sum(residues);
% r_m=sum(residues.*w)/sum(w);
% w=rorWeightsUpdate2(residues, w);

% v=max(residues);
% v=5

while(d>10^-5)
% while(max(residues.*w)>10)
% while(max(residues.*w)>10)
    t=t+1;
%     if max(residues.*w)<10
%         break
%     end


%     w=rorWeightsUpdate2(residues, w);
%     if v>10
%     v=v/2;
%     end
%     w=w/max(w)
    w=rorWeightsUpdate(residues, w);

%     if sum(w)<=0.05*numel(w) || (max(residues.*w)<10)
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
% result.R-lR_gt;
tran_error=norm(result.t-lt_gt');
end

function [residues]=r(a,P,RR,tt,noise)
    
    residues=((RR*a+tt)-P).^2;
    residues=sum(residues,1);
    residues=residues/(noise^2);

end


function [weights] = rorWeightsUpdate(residuals,weights)

lam=zeros(1,length(residuals));

for k = 1:length(residuals)
lam(k)=residuals(k);
weights(k)=(5+3)/(lam(k)+5);
end
end
