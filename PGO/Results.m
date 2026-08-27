close all
MC=2;
outlier_max=9;
T = table('Size',[5*MC*outlier_max 4],'VariableTypes',{'double','double','double','double'});

% data_file='ResTable_CSAIL.txt';
data_file='ResTable_intel.txt';


fid = fopen(data_file, 'r');
read_line = fgets(fid);  % Read the next line from the file
ii=0;
while ischar(read_line)  % As long as this line is a valid character string
        ii=ii+1;
        [T.Var1(ii), T.Var2(ii), T.Var3(ii), T.Var4(ii)] = strread(read_line, '%f %f %f %f');
        read_line = fgets(fid);
end



skip=0;
T((1:MC*5*skip),:) = [];
T.Var3=categorical(T.Var3);


figure
boxchart(T.Var3,T.Var1,'GroupByColor',T.Var2)
xline(1.5:1:10.5)
ax = gca;
ax.Box = 'on';
set(ax,'FontSize',16);
% ax.YAxis.Scale ="log";
xlabel('Outlier Ratio','FontSize',24,'Interpreter','latex');
ylabel('RMSE (m)','FontSize',24,'Interpreter','latex');
legend('GNS-GM','GNS-TLS','EROR','ESOR','ASOR','NumColumns',2,'Location','NorthWest')
% ax.YAxis.Scale ="log";
grid on
ylim([0 8])
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 7.5, 5], 'PaperUnits', 'Inches')
saveas(gcf,'intel_rmse','epsc')
% saveas(gcf,'CSAIL_rmse','epsc')

figure
boxchart(T.Var3,T.Var4,'GroupByColor',T.Var2)
xline(1.5:1:10.5)
% arrayfun(@(x)xline(x,'r-','LineWidth',.5),T.Var3)
legend('GNS-GM','GNS-TLS','EROR','ESOR','ASOR','NumColumns',2,'Location','NorthWest')
ax = gca;
ax.Box = 'on';
set(ax,'FontSize',16);
ax.YAxis.Scale ="log";
grid on
xlabel('Outlier Ratio','FontSize',24,'Interpreter','latex');
ylabel('Time (s)','FontSize',24,'Interpreter','latex');
ylim([0 3e3])
% 
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 7.5, 5], 'PaperUnits', 'Inches')
% saveas(gcf,'CSAIL_time','epsc')
saveas(gcf,'intel_time','epsc')

% 

