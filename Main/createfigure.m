function createfigure(X1, YMatrix1)
%CREATEFIGURE(X1, YMATRIX1)
%  X1:  x ���ݵ�ʸ��
%  YMATRIX1:  y ���ݵľ���

%  �� MATLAB �� 13-May-2018 11:13:12 �Զ�����

% ���� figure
figure1 = figure;

% ���� axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% ʹ�� plot �ľ������봴������
plot1 = plot(X1,YMatrix1,'Marker','*','Parent',axes1);
set(plot1(1),'DisplayName','���ʺ���',...
    'Color',[0.929411768913269 0.694117665290833 0.125490203499794]);
set(plot1(2),'DisplayName','��϶��',...
    'Color',[0.0705882385373116 0.211764708161354 0.141176477074623]);
set(plot1(3),'DisplayName','��͸��','Color',[1 0 0]);
set(plot1(4),'DisplayName','�������Ͷ�',...
    'Color',[0.301960796117783 0.745098054409027 0.933333337306976]);

% ���� xlabel
xlabel('���Լ��������');

% ���� title
title('ELM���Լ�Ԥ�����');

% ���� ylabel
ylabel('�������');

box(axes1,'on');
% ���� legend
legend(axes1,'show');

