% This is the source code of paper : 'Bayesian Heuristics for Robust Spatial Perception' (chughtaiah@gmail.com) for mesh regitration
% The authors acknowledge the authors of Convex Global 3D Registration with
% Lagrangian Duality (CVPR 17) https://github.com/jbriales/CVPR17
% main_mesh.m file compares performance of GNC-TLS, GNC-GM, EROR, ESOR, ASOR

example
clear;
clc
close all

%%%%load data files by providing local path
load('data\car.mat');
% load('data\motorbike.mat');


vertices=car(1).vertices;
lines_vertices=car(1).faces;
faces=car(1).faces;

lines_vertices=[lines_vertices(:,1:2);lines_vertices(:,2:3)];

for i=1:size(lines_vertices,1)
    if(lines_vertices(i,1)>lines_vertices(i,2))
        temp=lines_vertices(i,1);
        lines_vertices(i,1)=lines_vertices(i,2);
        lines_vertices(i,2)=temp;
    end
end

lines_vertices=unique(lines_vertices(:,1:2),'rows');

problem.m = [100 80 80]; 
problem.noise = 0.01;

%%set MC runs; can be set to lower number to gauge results 
MC=10;
outlier_max=9;
T = table('Size',[5*MC*outlier_max 5],'VariableTypes',{'double','double','string','double','double'});

warning('off','all')

% for ii=9
for ii=1:outlier_max
    ii

GNC_GM_a=zeros(MC,1);    
GNC_GM_tr=zeros(MC,1);    
GNC_GM_time=zeros(MC,1);   

GNC_TLS_a=zeros(MC,1);    
GNC_TLS_tr=zeros(MC,1);    
GNC_TLS_time=zeros(MC,1);    

EROR_a=zeros(MC,1);    
EROR_tr=zeros(MC,1);    
EROR_time=zeros(MC,1);    

ESOR_a=zeros(MC,1);    
ESOR_tr=zeros(MC,1);    
ESOR_time=zeros(MC,1); 

ASOR_a=zeros(MC,1);    
ASOR_tr=zeros(MC,1);    
ASOR_time=zeros(MC,1); 

outlier_ratio=.1*ii;


for j=1:MC
    ii
    j

exc=1;

while(exc==1)

try

[correspondences,gt_T] = mesh_registration( problem.m, problem.noise ,vertices, lines_vertices, faces, outlier_ratio*100); 
    
display('GNC_TLS')
start_time=now;
[GNC_TLS_a(j),GNC_TLS_tr(j)]=GNC_TLS_mesh(correspondences,problem.noise,gt_T);
end_time=now;
GNC_TLS_time(j)=(end_time-start_time)*24*60*60;

display('EROR')
start_time=now;
[EROR_a(j),EROR_tr(j)]=EROR_mesh(correspondences,problem.noise,gt_T);
end_time=now;
EROR_time(j)=(end_time-start_time)*24*60*60;

display('ASOR')
start_time=now;
[ASOR_a(j),ASOR_tr(j)]=ASOR_mesh(correspondences,problem.noise,gt_T);
end_time=now;
ASOR_time(j)=(end_time-start_time)*24*60*60;

display('ESOR')
start_time=now;
[ESOR_a(j),ESOR_tr(j)]=ESOR_mesh(correspondences,problem.noise,gt_T);
end_time=now;
ESOR_time(j)=(end_time-start_time)*24*60*60;

display('GNC_GM')
start_time=now;
[GNC_GM_a(j),GNC_GM_tr(j)]=GNC_GM_mesh(correspondences,problem.noise,gt_T);
end_time=now;
GNC_GM_time(j)=(end_time-start_time)*24*60*60;

exc=0;
catch
display('exception')    
exc=1;    
end

end

end

T.Var1((ii-1)*MC*5+1:ii*MC*5,1)=[GNC_GM_a;GNC_TLS_a;EROR_a;ESOR_a;ASOR_a];
T.Var2((ii-1)*MC*5+1:ii*MC*5,1)=[GNC_GM_tr;GNC_TLS_tr;EROR_tr;ESOR_tr;ASOR_tr];
T.Var3((ii-1)*MC*5+1:ii*MC*5,1)=[repmat(0,MC,1);repmat(1,MC,1);repmat(2,MC,1);repmat(3,MC,1);repmat(4,MC,1)];
T.Var4((ii-1)*MC*5+1:ii*MC*5,1)=repmat((outlier_ratio),MC*5,1);
T.Var5((ii-1)*MC*5+1:ii*MC*5,1)=[GNC_GM_time;GNC_TLS_time;EROR_time;ESOR_time;ASOR_time];

end

skip=0;
T((1:MC*5*skip),:) = [];
T.Var4=categorical(T.Var4);

figure
boxchart(T.Var4,T.Var1,'GroupByColor',T.Var3)
xline(1.5:1:10.5)
ax = gca;
ax.Box = 'on';
set(ax,'FontSize',16);
ax.YAxis.Scale ="log";
xlabel('Outlier Ratio','FontSize',24,'Interpreter','latex');
ylabel('Rotation Error (deg)','FontSize',24,'Interpreter','latex');
legend('GNC-GM','GNC-TLS','EROR','ESOR','ASOR','NumColumns',2,'Location','NorthWest')
grid on
ylim([0 3e2])
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 7.5, 5], 'PaperUnits', 'Inches')
saveas(gcf,'meshro_01','epsc')


figure
boxchart(T.Var4,T.Var2,'GroupByColor',T.Var3)
xline(1.5:1:10.5)
ax = gca;
ax.Box = 'on';
set(ax,'FontSize',16);
ax.YAxis.Scale ="log";
xlabel('Outlier Ratio','FontSize',24,'Interpreter','latex');
ylabel('Translation Error (m)','FontSize',24,'Interpreter','latex');
legend('GNC-GM','GNC-TLS','EROR','ESOR','ASOR','NumColumns',2,'Location','NorthWest')
grid on
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 7.5, 5], 'PaperUnits', 'Inches')
saveas(gcf,'meshtr_01','epsc')


figure
boxchart(T.Var4,T.Var5,'GroupByColor',T.Var3)
xline(1.5:1:10.5)
% arrayfun(@(x)xline(x,'r-','LineWidth',.5),T.Var3)
legend('GNC-GM','GNC-TLS','EROR','ESOR','ASOR','NumColumns',2,'Location','NorthWest')
ax = gca;
ax.Box = 'on';
set(ax,'FontSize',16);
ax.YAxis.Scale ="log";
grid on
xlabel('Outlier Ratio','FontSize',24,'Interpreter','latex');
ylabel('Time (s)','FontSize',24,'Interpreter','latex');
% ylim([0 30])
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 7.5, 5], 'PaperUnits', 'Inches')
saveas(gcf,'meshti_01','epsc')


