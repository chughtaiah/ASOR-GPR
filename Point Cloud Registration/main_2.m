% This is the source code of paper : 'Bayesian Heuristics for Robust Spatial Perception' (chughtaiah@gmail.com)
% The authors acknowledge the authors of 'RANSIC: Fast and Highly Robust Estimation for Rotation Search and Point Cloud Registration using Invariant Compatibility 
% that inspired this code and the Environment routine
% and  Matt Jacobson, http://www.xorantech.com for the standard open-source horns.m function
%main_2.m file compares performance of GNC-TLS, GNC-GM, EROR, ESOR, ASOR and standard SOR, ROR and Least-Square solvers
 

clc;
clear;
close all;

%% Parameter Setup

known_scale=1; % '0' for unknown scale and '1' for known scale
% set outlier ratio in percentage
show_figure=0; % '1' for displaying and '0' for not displaying


%%


% number of points
n_ele=100;


% set nominal noise levels
noise=.01;
% 


% scale is within 1-5 (unknown scale) or 1 (known scale)
if known_scale==1 %known scale
scale_gt=1;
elseif known_scale==0 % unknown scale
scale_gt=1+4*rand(1);
end


%% Set MC runs (MC)and max percentage of outlier (outlier_max)
MC=20;
outlier_max=9;


T = table('Size',[5*MC*outlier_max 5],'VariableTypes',{'double','double','double','double','double'});


for ii=1:outlier_max
% for ii=0   
ii
GNC_TLS_a=zeros(MC,1);    
GNC_TLS_tr=zeros(MC,1);    
GNC_TLS_time=zeros(MC,1);    

GNC_GM_a=zeros(MC,1);    
GNC_GM_tr=zeros(MC,1);    
GNC_GM_time=zeros(MC,1);    
%   

EROR_a=zeros(MC,1);    
EROR_tr=zeros(MC,1);    
EROR_time=zeros(MC,1);    

ESOR_a=zeros(MC,1);    
ESOR_tr=zeros(MC,1);    
ESOR_time=zeros(MC,1);


ROR_a=zeros(MC,1);    
ROR_tr=zeros(MC,1);    
ROR_time=zeros(MC,1);    

SOR_a=zeros(MC,1);    
SOR_tr=zeros(MC,1);    
SOR_time=zeros(MC,1);

LS_a=zeros(MC,1);    
LS_tr=zeros(MC,1);    
LS_time=zeros(MC,1);

ASOR_a=zeros(MC,1);    
ASOR_tr=zeros(MC,1);    
ASOR_time=zeros(MC,1); 

outlier_ratio=.1*ii;
for j=1:MC
j
   

% build environment
[pts_3d,pts_3d_,R_gt,t_gt]=Environment(n_ele,noise,outlier_ratio,scale_gt,show_figure);


tic
[GNC_GM_a(j),GNC_GM_tr(j)]=GNS_GM(pts_3d,pts_3d_,n_ele,R_gt,t_gt,noise);
GNC_GM_time(j)=toc;

tic
[GNC_TLS_a(j),GNC_TLS_tr(j)]=GNS_TLS(pts_3d,pts_3d_,n_ele,R_gt,t_gt,noise);
GNC_TLS_time(j)=toc;


tic
[EROR_a(j),EROR_tr(j)]=EROR(pts_3d,pts_3d_,n_ele,R_gt,t_gt,noise);
EROR_time(j)=toc;

tic
[ESOR_a(j),ESOR_tr(j)]=ESOR(pts_3d,pts_3d_,n_ele,R_gt,t_gt,noise);
ESOR_time(j)=toc;

tic
[ASOR_a(j),ASOR_tr(j)]=ASOR(pts_3d,pts_3d_,n_ele,R_gt,t_gt,noise);
ASOR_time(j)=toc;


tic
[ROR_a(j),ROR_tr(j)]=ROR(pts_3d,pts_3d_,n_ele,R_gt,t_gt,noise);
ROR_time(j)=toc;

tic
[SOR_a(j),SOR_tr(j)]=SOR(pts_3d,pts_3d_,n_ele,R_gt,t_gt,noise);
SOR_time(j)=toc;

tic
[LS_a(j),LS_tr(j)]=LS(pts_3d,pts_3d_,n_ele,R_gt,t_gt,noise);
LS_time(j)=toc;

end

T.Var1((ii-1)*MC*8+1:ii*MC*8,1)=[GNC_GM_a;GNC_TLS_a;EROR_a;ESOR_a;ASOR_a;SOR_a;ROR_a;LS_a];
T.Var2((ii-1)*MC*8+1:ii*MC*8,1)=[GNC_GM_tr;GNC_TLS_tr;EROR_tr;ESOR_tr;ASOR_tr;SOR_tr;ROR_tr;LS_tr];
T.Var3((ii-1)*MC*8+1:ii*MC*8,1)=[repmat(0,MC,1);repmat(1,MC,1);repmat(2,MC,1);repmat(3,MC,1);repmat(4,MC,1);repmat(5,MC,1);repmat(6,MC,1);repmat(7,MC,1)];
T.Var4((ii-1)*MC*8+1:ii*MC*8,1)=repmat((outlier_ratio),MC*8,1);
T.Var5((ii-1)*MC*8+1:ii*MC*8,1)=[GNC_GM_time;GNC_TLS_time;EROR_time;ESOR_time;ASOR_time;SOR_time;ROR_time;LS_time];

end
skip=0;
T((1:MC*4*skip),:) = [];
T((1:MC*5*skip),:) = [];
T.Var4=categorical(T.Var4);

% f=figure
% f.Position = [10 10 550 400]; 
figure
b=boxchart(T.Var4,T.Var1,'GroupByColor',T.Var3);
b(6,1).BoxFaceColor = [0 0 0];
b(7,1).BoxFaceColor = [0 1 0];
b(8,1).BoxFaceColor = [1 0 1];

xline(1.5:1:10.5)
ax = gca;
ax.Box = 'on';
set(ax,'FontSize',16);
ax.YAxis.Scale ="log";
xlabel('Outlier Ratio','FontSize',24,'Interpreter','latex');
ylabel('Rotation Error (deg)','FontSize',24,'Interpreter','latex');
legend('GNS-GM','GNS-TLS','EROR','ESOR','ASOR','SOR','ROR','LS','NumColumns',2,'Location','NorthWest')
grid on
ylim([0 5*10^2])
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 7.5, 5], 'PaperUnits', 'Inches')
% saveas(gcf,'hornsro_01_N100_arm','epsc')
% saveas(gcf,'hornsro_01_N100','epsc')


figure
b=boxchart(T.Var4,T.Var2,'GroupByColor',T.Var3);
b(6,1).BoxFaceColor = [0 0 0];
b(7,1).BoxFaceColor = [0 1 0];
b(8,1).BoxFaceColor = [1 0 1];
xline(1.5:1:10.5)
ax = gca;
ax.Box = 'on';
set(ax,'FontSize',16);
ax.YAxis.Scale ="log";
xlabel('Outlier Ratio','FontSize',24,'Interpreter','latex');
ylabel('Translation Error (m)','FontSize',24,'Interpreter','latex');
legend('GNS-GM','GNS-TLS','EROR','ESOR','ASOR','SOR','ROR','LS','NumColumns',2,'Location','NorthWest')
grid on
ylim([0 2*10^0])
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 7.5, 5], 'PaperUnits', 'Inches')
% saveas(gcf,'hornstr_01_N100_arm','epsc')
% saveas(gcf,'hornstr_01_N100','epsc')


figure
b=boxchart(T.Var4,T.Var5,'GroupByColor',T.Var3);
b(6,1).BoxFaceColor = [0 0 0];
b(7,1).BoxFaceColor = [0 1 0];
b(8,1).BoxFaceColor = [1 0 1];
xline(1.5:1:10.5)
legend('GNS-GM','GNS-TLS','EROR','ESOR','ASOR','SOR','ROR','LS','NumColumns',2,'Location','NorthWest')
ax = gca;
ax.Box = 'on';
set(ax,'FontSize',16);
ax.YAxis.Scale ="log";
grid on
xlabel('Outlier Ratio','FontSize',24,'Interpreter','latex');
ylabel('Time (s)','FontSize',24,'Interpreter','latex');
% ylim([0 20*10^-3])
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 7.5, 5], 'PaperUnits', 'Inches') 
% saveas(gcf,'hornsti_01_N100_arm','epsc')
% saveas(gcf,'hornsti_01_N100','epsc')



