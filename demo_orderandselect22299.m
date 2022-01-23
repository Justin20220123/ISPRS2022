clc
clear ;
close all ;


%% according to visually standard image getting 读取并显示类标图像和类标总数
image=imread('E:\20200430Experiment\POLSAR Images\ESAR_Oberpfaffenhofen1200-1300\standard image\标准图.bmp');
figure(),imshow(uint8(image)) 
[r,c,gg]=size(image);

%% 
addpath 'E:\20200430Experiment\AE_with_finetune\';
[val, Numval,indexing] =Findvalue(image); %找到图像中，像素值，类别，个数信息并建立索引indexing
K=size(indexing,1);%总类别数
Labels=Visual2Labels(image,indexing);%可视化的类标图像根据索引indexing转换为标签矩阵Labels

%% obtain Polar_Decompose features
path='E:\20200430Experiment\POLSAR Images\ESAR_Oberpfaffenhofen1200-1300\TouziWindows\17\T3\';
[DATA_Feat, DATA_Feat_order]=Touzi_Decomp_ALLPara_ESAR(path,r,c);





%% Paras.
DATA=DATA_Feat;

[Dim,~]=size(DATA);
inputSize =Dim;              % Inputlayer 
numClasses = K-1;           % Outputlayer  Layer 3
softmaxLambda = 1e-8;

Ntimes=30;
alpha=0.01;
%% demo function1
DATA1=DATA;      % ensure the data invariable
addpath 'E:\20200430Experiment\data_preprocessing\'
DATA1=ZCA_Whitening(DATA1);
[Dim1,~]=size(DATA1);

addpath 'E:\20200430Experiment\'
for nn=1:Dim1
    for kkk=1:Ntimes
        [pred1ZA(kkk,:),acc1ZA(kkk,nn)]=Softmax_Classifier_JustinESAR(nn,numClasses,softmaxLambda,DATA1(1:nn,:),Labels,K,alpha);
        nn
        disp('输出字符串！!!!!!!**********************************\n%%%%$$$$$$$$') 
    end
end

%%
figure()
plot(1:Dim1,acc1ZA,'--gd',1:Dim1,mean(acc1ZA,1),'p')

x1=1:1:Dim1;
cell_1={'\lambda_1','\lambda_2','\lambda_3','SPAN',...
    '\alpha_1','\alpha_2','\alpha_3','\Phi_1', '\Phi_2', '\Phi_3', ...
    '\Psi_1','\Psi_2','\Psi_3', '\tau_1','\tau_2','\tau_3'};%分类方法
set(gca,'xtick',x1);
set(gca,'xticklabel',cell_1);


%% demo function2
DATA_order=DATA_Feat_order;

DATA2=DATA_order;
DATA2=ZCA_Whitening(DATA2);
[Dim2,~]=size(DATA2);

addpath 'E:\20200430Experiment\'
for nn=1:Dim2
    for kkk=1:Ntimes
        [pred1ZP(kkk,:),acc1ZP(kkk,nn)]=Softmax_Classifier_JustinESAR(nn,numClasses,softmaxLambda,DATA2(1:nn,:),Labels,K,alpha);
        nn
        disp('输出字符串！!!!!!!**********************************\n%%%%$$$$$$$$') 
    end
end


%%
figure()
H=plot(1:Dim2,acc1ZP,'-r*',1:Dim2,mean(acc1ZP,1),'-bd')

x2=1:1:Dim2;
cell_2={'\lambda_1','\lambda_2','\lambda_3','SPAN',  ...
    '\alpha_{s1}','\alpha_{s2}','\alpha_{s3}','\Phi_\alpha_{s1}', '\Phi_\alpha_{s2}', '\Phi_\alpha_{s3}',...
    '\Psi_1','\Psi_2','\tau_1','\tau_2','\Psi_3','\tau_3'};%分类方法
set(gca,'xtick',x2);
set(gca,'xticklabel',cell_2);
xlabel('Touzi Parameters');
ylabel('OA');
% set(gca,'xtick',0:1:Ntimes);  
set(gca,'ytick',0:0.1:1);  


set(gca,'FontSize',12);    % 设置坐标轴的数字大小，包括legend文字大小：
% legend('Eachiter','MeanValue','MeanValue1','MeanValue2','Location','Best')  % 各曲线线型的说明信息，位置：右下角
h1=legend(H([1 11 25 30 31]),'1','11','25','30','31','Location','SouthEast')
%使得标注横向显示
set(h1,'Orientation','horizon')
%去除标记框的外围长方形
set(h1,'Box','off');

xtickangle(-90)%  controll the rotation angle of xlabel
ytickangle(-30)%  controll the rotation angle of ylabel

saveas(gcf,'Order_ESAR.bmp')


%%

%% 随机选取定量的维度的特征测试特征数量对精度的影响
%% demo function1
num=10; %huozhe 13
DATA3=DATA;      % ensure the data invariable
addpath 'E:\20200430Experiment\data_preprocessing\'
DATA3=ZCA_Whitening(DATA3);
[Dim3,~]=size(DATA3);
uu=randperm(Dim3);
vv=uu(:,1:num);
DATA4=DATA3(vv,:);

for kkk=1:Ntimes
    [pred11_RZP(kkk,:),acc11_RZP(kkk)]=Softmax_Classifier_JustinESAR(num,numClasses,softmaxLambda,DATA4,Labels,K,alpha);
    disp('输出根据特征顺序随机选取的特征个数的结果\n%%%%$$$$$$$$')
end

%%  #Features>7 以后，我们根据图像包含的信息和直方图曲线选取的最优#Features=13;展示结果
DATA5=[DATA_order(1:6,:);DATA_order(8:9,:);DATA_order(11:12,:)];
[Dim5,~]=size(DATA5);
addpath 'E:\20200430Experiment\data_preprocessing\'
DATA5=ZCA_Whitening(DATA5);
for kkk=1:Ntimes
    [pred11_SZP(kkk,:),acc11_SZP(kkk)]=Softmax_Classifier_JustinESAR(Dim5,numClasses,softmaxLambda,DATA5,Labels,K,alpha);
    disp('输出根据直方图和特征参数图构成的特征顺序选取的最优特征个数的结果\n%%%%$$$$$$$$')
end


%%  对所有的特征值进行ZCA预处理然后进行直接分类;展示结果0.8489

for kkk=1:Ntimes
    [pred11_AZP(kkk,:),acc11_AZP(kkk)]=Softmax_Classifier_JustinESAR(Dim,numClasses,softmaxLambda,DATA1,Labels,K,alpha);
    disp('输出所有特征的结果\n%%%%$$$$$$$$')
end


%%
figure()
plot(1:Ntimes,acc11_RZP,'-r*',1:Ntimes,acc11_SZP,'-gd',1:Ntimes,acc11_AZP,'-bo')
hold on
plot(1:Ntimes,repmat(mean(acc11_RZP),1,Ntimes),'-.r',1:Ntimes,repmat(mean(acc11_SZP),1,Ntimes),'-.g',1:Ntimes,repmat(mean(acc11_AZP),1,Ntimes),'-.b')
xlabel('Iters');
ylabel('OA');
set(gca,'xtick',0:2:Ntimes);  
set(gca,'ytick',0:0.01:1.0);  
% set(gca,'ytick',0:0.1:1);  


set(gca,'FontSize',12);    % 设置坐标轴的数字大小，包括legend文字大小：
legend('RP','SP','AP','MRP','MSP','MAP','Location','Best')  % 各曲线线型的说明信息，位置：右下角

xtickangle(45)%  controll the rotation angle of xlabel



saveas(gcf,'Select_ESAR.bmp')

%%


% save orderandselect_ESAR.mat