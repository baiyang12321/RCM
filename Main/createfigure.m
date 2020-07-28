function createfigure(X1, YMatrix1)
%CREATEFIGURE(X1, YMATRIX1)
%  X1:  x 数据的矢量
%  YMATRIX1:  y 数据的矩阵

%  由 MATLAB 于 13-May-2018 11:13:12 自动生成

% 创建 figure
figure1 = figure;

% 创建 axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% 使用 plot 的矩阵输入创建多行
plot1 = plot(X1,YMatrix1,'Marker','*','Parent',axes1);
set(plot1(1),'DisplayName','泥质含量',...
    'Color',[0.929411768913269 0.694117665290833 0.125490203499794]);
set(plot1(2),'DisplayName','孔隙度',...
    'Color',[0.0705882385373116 0.211764708161354 0.141176477074623]);
set(plot1(3),'DisplayName','渗透率','Color',[1 0 0]);
set(plot1(4),'DisplayName','含气饱和度',...
    'Color',[0.301960796117783 0.745098054409027 0.933333337306976]);

% 创建 xlabel
xlabel('测试集样本编号');

% 创建 title
title('ELM测试集预测误差');

% 创建 ylabel
ylabel('绝对误差');

box(axes1,'on');
% 创建 legend
legend(axes1,'show');

