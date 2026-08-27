clear;
clc


% load('data\motorbike.mat');
load('data\car.mat');

mesh_bike = extendedObjectMesh(car(1).vertices,car(1).faces);
% mesh_bike = extendedObjectMesh(boat(1).vertices,boat(1).faces);

% mesh_bike = extendedObjectMesh(car(2).vertices,car(2).faces);

% mesh_bike = extendedObjectMesh(motorbike(1).vertices,motorbike(1).faces);

% pcshow(motorbike(1).vertices);
a=show(mesh_bike);
properties(a)
a.Color=[.5 .5 .5];
% hold on
% % pcshow(motorbike(1).vertices,'Markersize',10)
% pcshow(motorbike(1).vertices,'Markersize',10)


% vertices=motorbike(1).vertices;
% lines_vertices=motorbike(1).faces;
% faces=motorbike(1).faces;

vertices=car(2).vertices;
lines_vertices=car(2).faces;
faces=car(2).faces;

lines_vertices=[lines_vertices(:,1:2);lines_vertices(:,2:3)];
for i=1:size(lines_vertices,1)
    if(lines_vertices(i,1)>lines_vertices(i,2))
        temp=lines_vertices(i,1);
        lines_vertices(i,1)=lines_vertices(i,2);
        lines_vertices(i,2)=temp;
    end
end
lines_vertices=unique(lines_vertices(:,1:2),'rows');


 problem.m = [100 10 10]; 
    % noise in the correspondences
 problem.noise = 0.05*.2;
    % size of the random scene




numObs=problem.m;
sd=problem.noise;
percent=.0*100;
% Generate random set of plane correspondences
p2p_correspondences = bike_pointToPoint(numObs(1),vertices);
p2l_correspondences = bike_pointToLine(numObs(2),vertices,lines_vertices);
p2pl_correspondences = bike_pointToPlane(numObs(3),vertices,faces);

% Concatenate all correspondences
correspondences = [p2p_correspondences, p2l_correspondences, p2pl_correspondences];

T = Pose.rand();
IT=inv(T);

% mesh_bike2 = extendedObjectMesh((motorbike(1).vertices*IT.R')+IT.t',motorbike(1).faces);
% show(mesh_bike2)
hold on
% pcshow((motorbike(1).vertices*IT.R')+IT.t','Markersize',50)
pcshow((car(1).vertices*IT.R')+IT.t','Markersize',50)

% 
% % Transform all sampled points into measured points (with inverse of GT)
allPoints = [correspondences.point];
pts_3d=plotter(allPoints);
transform(allPoints,inv(T));

crp_pnts1 = floor(percent*numObs(1)/100);
ind1=randperm(crp_pnts1);

crp_pnts2 = floor(percent*numObs(2)/100);
ind2=[1+numObs(1):numObs(1)+crp_pnts2];
i2=ind2(randperm(length(ind2)));


crp_pnts3 = floor(percent*numObs(3)/100);
ind3=[1+numObs(1)+numObs(2)+1:1+numObs(1)+numObs(2)+crp_pnts3];
i3=ind3(randperm(length(ind3)));

color_index=zeros(1,numObs(1)+numObs(2)+numObs(3));

for i=1:numObs(1)+numObs(2)+numObs(3)
  corrupt(correspondences(i).point,sd);   
end

for i=1:crp_pnts1
tmp(i)=correspondences(i).point;
color_index(i)=1;% outlier for 1
end

for i=1:crp_pnts1
  correspondences(ind1(i)).point=tmp(i);
end

for i=1+numObs(1):numObs(1)+crp_pnts2
tmp(i)=correspondences(i).point;
color_index(i)=1;
end

for i=1+numObs(1):numObs(1)+crp_pnts2
  correspondences(i2(i-numObs(1))).point=tmp(i);
end

for i=1+numObs(1)+numObs(2)+1:1+numObs(1)+numObs(2)+crp_pnts3
  tmp(i)=correspondences(i).point;
  color_index(i)=1;
end

for i=1+numObs(1)+numObs(2)+1:1+numObs(1)+numObs(2)+crp_pnts3
  correspondences(i3(i-(numObs(1)+numObs(2)+1))).point=tmp(i);
end

allPoints = [correspondences.point];
pts_3d_=plotter(allPoints);

for i=1:numObs(1)+numObs(2)+numObs(3)
    
    if color_index(i)==0
    plot3([pts_3d(i,1),pts_3d_(i,1)],[pts_3d(i,2),pts_3d_(i,2)],[pts_3d(i,3),pts_3d_(i,3)],'b','LineWidth',1);
    else
    pp=plot3([pts_3d(i,1),pts_3d_(i,1)],[pts_3d(i,2),pts_3d_(i,2)],[pts_3d(i,3),pts_3d_(i,3)],'r','LineWidth',1);
%     pp.Color(4) = 0.15;
    end
    
    set(gcf,'color','w');
    grid off;
    axis off;

end



function correspondences = bike_pointToPoint(m,vert)

% preallocate objects
correspondences = repmat(Point2Point(),1,m);
perm=randperm(size(vert,1),m);
test=[];
for j=1:m % for each correspondence
  i=perm(j);
  % Choose plane randomly such that it intersects the sphere
  % centered around zero with radius R.
  p1 = Point(vert(i,:)');
  % The corresponding point is the same point
  p2 = copy(p1);
  test=[test;vert(i,:)];
  % store new object
  correspondences(j) = Point2Point(p2,p1);
end
% pcshow(test,'Markersize',50)
end

function correspondences = bike_pointToLine(m,vert,lin_vert)

% preallocate objects
correspondences = repmat(Point2Line(),1,m);

test=[];
size(lin_vert,1);
perm=randperm(size(lin_vert,1),m);

% i = randperm(numel(lin_vert),m);
for j=1:m % for each correspondence
  i=perm(j);
  % Choose plane randomly such that it intersects the sphere
  % centered around zero with radius R.
  vect=vert(lin_vert(i,2),:)-vert(lin_vert(i,1),:);
  line = Line(vert(lin_vert(i,1),:)',vect'/norm(vect));
  
  % Sample a random point in the line
  % 1. Generate random 3D point inside the sphere
  pnt = vert(lin_vert(i,1),:) + (vert(lin_vert(i,2),:)-vert(lin_vert(i,1),:))*rand;
  point=Point(pnt');
  % 2. Project point into line
  test=[test;pnt];
  % store new object
  correspondences(j) = Point2Line(point,line);
end
% pcshow(test,'Markersize',50)
end

function correspondences = bike_pointToPlane(m,vert,face)
% 
% preallocate objects
correspondences = repmat(Point2Plane(),1,m);
test=[];
perm=randperm(size(face,1),m);

for j=1:m % for each correspondence
  % Choose plane randomly such that it intersects the sphere
  % centered around zero with radius R.\
  i=perm(j);
  normal=cross( vert(face(i,2),:)-vert(face(i,1),:),vert(face(i,3),:)-vert(face(i,1),:) ) ;
  plane = Plane(vert(face(i,1),:)',snormalize(normal'));
  
  % 2. Project point into plane
  pnt = vert(face(i,1),:) + (vert(face(i,2),:)-vert(face(i,1),:))*rand;
  pnt = pnt + (vert(face(i,3),:)-pnt )*rand;
  test=[test;pnt];
  
  point=Point(pnt');
  % store new object
  correspondences(j) = Point2Plane(point,plane);
end
% pcshow(test,'Markersize',50)
end
 
function tst=plotter(pts)
hold on
tst=[];
for i=1:numel(pts)
tst=[tst;pts(1,i).x'];   
end
pcshow(tst,'Markersize',5)
end

function plot_mesh(pts)
tst=[];
for i=1:numel(pts)
tst=[tst;pts(i,:)];   
end
pcshow(tst,'Markersize',5)
end