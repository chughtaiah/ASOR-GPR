function [pts_3d,pts_3d_,R,t]=Environment(n_ele,noise,outlier_ratio,scale_gt,show_figure)
%% select dataset

pcd = pcread('data/bunny.ply');
%pcd = pcread('data/Armadillo.ply');

% pcd_down = pcdownsample(pcd, 'random', 0.029); 

pcd_down = pcdownsample(pcd, 'random', 1);

pts_3d = 6*pcd_down.Location;

pts_3d=double(pts_3d(1:n_ele,:));

pts_3d=pts_3d-0.5*[max(pts_3d(:,1))+min(pts_3d(:,1)), max(pts_3d(:,2))+min(pts_3d(:,2)), max(pts_3d(:,3))+min(pts_3d(:,3))];
pts_3d=(pts_3d)./ [2*max(pts_3d(:,1)), 2*max(pts_3d(:,2)), 2*max(pts_3d(:,3))] ;


n_pts=n_ele;

%% Transformation
%Generate random rotation
axis1= rand(1,3)-0.5;

axis1=axis1/norm(axis1);

angle=2*pi*rand(1);

aa = angle * axis1;

if norm(aa) < 2e-16
         R=eye(3);
else k = aa / norm(aa);
K1=[0, -k(3), k(2); k(3), 0, -k(1); -k(2), k(1), 0];

%construct a rotation matrix from an axis angle representation
R = eye(3) + sin(norm(aa)) * K1 + (1 -cos(norm(aa))) * K1 * K1;
end

%Generate random translation

t_raw=rand(1,3)-0.5;

t=0.0*[1,1,1]+(3*rand(1))*t_raw/norm(t_raw);

%transform by R & t
d_med= scale_gt * pts_3d * R';

    for i=1:n_pts
    pts_(i,:) = d_med(i,:) + t ;
    end

%add noise

pts_3d_=pts_+normrnd(0,noise,n_ele,3);

%add outliers

for i=1:round(n_ele*outlier_ratio)
    
    for iii=1:1e+18
    rand_vec=2*scale_gt*rand(1,3)-scale_gt;
        if norm(rand_vec)<=0.5*sqrt(3)*scale_gt
            break
        end
    end
    
pts_3d_(i,:)=[0.0,0.0,0.0]+t+rand_vec;%5*mean(pts_3d_(i,:))*rand(1,3);

end


%% show figure

if show_figure==1
    
figure(1);

pc1=pointCloud(pts_3d(1:n_ele,:));
pc2_out=pointCloud(pts_3d_(1:round(outlier_ratio*n_ele),:));
pc2_in=pointCloud(pts_3d_(1+round(outlier_ratio*n_ele):n_ele,:));
pc2_=pointCloud(pts_(1:n_ele,:),'Color',ones(size(pts_(1:n_ele,:))).*[1 0 0]);

pcshow([pc1.Location(:,1),pc1.Location(:,2),pc1.Location(:,3)],[.07 .62 1]/1,'MarkerSize',60);

hold on;

pcshow([pc2_out.Location(:,1),pc2_out.Location(:,2),pc2_out.Location(:,3)],[1 0 0],'MarkerSize',10);

hold on;

% pcshow([pc2_in.Location(:,1),pc2_in.Location(:,2),pc2_in.Location(:,3)],[0 1 0],'MarkerSize',200);
pcshow([pc2_.Location(:,1),pc2_.Location(:,2),pc2_.Location(:,3)],[0 0 0],'MarkerSize',60);
% pcshow(pts_)

hold on;

for i=round(n_ele*outlier_ratio)+1:n_ele
    
    plot3([pts_3d(i,1),pts_3d_(i,1)],[pts_3d(i,2),pts_3d_(i,2)],[pts_3d(i,3),pts_3d_(i,3)],'b','LineWidth',.1);
    
end

for i=1:round(n_ele*outlier_ratio)
    
    pp=plot3([pts_3d(i,1),pts_3d_(i,1)],[pts_3d(i,2),pts_3d_(i,2)],[pts_3d(i,3),pts_3d_(i,3)],'r','LineWidth',.1);

    pp.Color(4) = 0.15;
    
end
% 
% set(gcf,'color','w');
% 
% grid off;
% axis off;

% title('Correspondences','FontSize',16,'Color','k');

end

end