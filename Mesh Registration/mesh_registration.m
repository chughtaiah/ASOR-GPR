function [correspondences,T] = mesh_registration( numObs, sd ,vertices,lines_vertices,faces,percent)

plot0=0; %1 on or 0 off

randMat = @() randn(3,numObs);

if isscalar(numObs)
  numObs = numObs * [1 1 1];
else assert(numel(numObs)==3)
end

% Generate random set of plane correspondences

p2p_correspondences = pointToPoint(numObs(1),vertices);
p2l_correspondences = pointToLine(numObs(2),vertices,lines_vertices);
[numObs(3),p2pl_correspondences] = pointToPlane(numObs(3),vertices,faces);

% Concatenate correspondences
correspondences = [p2p_correspondences, p2l_correspondences, p2pl_correspondences];

T = Pose.rand();
 
% Transform all sampled points into measured points (with inverse of GT)

allPoints = [correspondences.point];
pts_3d=plotter(allPoints,plot0);
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
pts_3d_=plotter(allPoints,plot0);

if(plot0==1)
for i=1:numObs(1)+numObs(2)+numObs(3)
    if color_index(i)==0
    plot3([pts_3d(i,1),pts_3d_(i,1)],[pts_3d(i,2),pts_3d_(i,2)],[pts_3d(i,3),pts_3d_(i,3)],'b','LineWidth',1);
    else
    pp=plot3([pts_3d(i,1),pts_3d_(i,1)],[pts_3d(i,2),pts_3d_(i,2)],[pts_3d(i,3),pts_3d_(i,3)],'r','LineWidth',1);
%     pp.Color(4) = 0.15;
    end
end
end


end

function correspondences = pointToPoint(m,vert)

% preallocate objects

correspondences = repmat(Point2Point(),1,m);
perm=randperm(size(vert,1),m);
test=[];
for j=1:m % for each correspondence
    i=perm(j);
  % Choose plane randomly such that it intersects the sphere % centered around zero with radius R.
  p1 = Point(vert(i,:)');
  % The corresponding point is the same point
  p2 = copy(p1);
  test=[test;vert(i,:)];
  % store new object
  correspondences(j) = Point2Point(p2,p1);
end
end

function correspondences = pointToLine(m,vert,lin_vert)

% preallocate objects
correspondences = repmat(Point2Line(),1,m);

test=[];
size(lin_vert,1);
perm=randperm(size(lin_vert,1),m);

for j=1:m % for each correspondence
    i=perm(j);
  % Choose plane randomly such that it intersects the sphere
  % centered around zero with radius R.
%   disp('line')
  vect=vert(lin_vert(i,2),:)-vert(lin_vert(i,1),:);
%   vert(lin_vert(i,1),:)',vect'/norm(vect);
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
end

function [valid_m,correspondences] = pointToPlane(m,vert,face)
% 
% preallocate objects
correspondences = repmat(Point2Plane(),1,m);
test=[];
perm=randperm(size(face,1),m);
valid_m=m;
k=0;
for j=1:m 
  % for each correspondence
  % Choose plane randomly such that it intersects the sphere
  % centered around zero with radius R.\
  i=perm(j);


  normal=cross( vert(face(i,2),:)-vert(face(i,1),:),vert(face(i,3),:)-vert(face(i,1),:) ); 
  nrmmm=snormalize(normal');
  if(sum(normal)==0)
      valid_m=valid_m-1;
      correspondences = repmat(Point2Plane(),1,valid_m);
      break
  else
  plane = Plane(vert(face(i,1),:)',snormalize(normal'));
  pnt = vert(face(i,1),:) + (vert(face(i,2),:)-vert(face(i,1),:))*rand;
  pnt = pnt + (vert(face(i,3),:)-pnt )*rand;
  test=[test;pnt];
  
  point=Point(pnt');
  % store new object
  k=k+1;  
  correspondences(k) = Point2Plane(point,plane);
  end

end
end

function tst=plotter(pts,plot0)
tst=[];
for i=1:numel(pts)
    tst=[tst;pts(1,i).x'];   
end
if(plot0==1)
    hold on
    pcshow(tst,'Markersize',50)
end
end