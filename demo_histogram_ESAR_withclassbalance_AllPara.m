clc
clear 
close all;
%% according to visually standard image getting 读取并显示类标图像和类标总数
image=imread('D:\20200430Experiment\POLSAR Images\ESAR_Oberpfaffenhofen1200-1300\standard image\标准图.bmp');
figure(),imshow(uint8(image)) 
[r,c,gg]=size(image);

%% 
addpath 'D:\20200430Experiment\AE_with_finetune\';
[val, Numval,indexing] =Findvalue(image); %找到图像中，像素值，类别，个数信息并建立索引indexing
K=size(indexing,1);%总类别数
Labels=Visual2Labels(image,indexing);%可视化的类标图像根据索引indexing转换为标签矩阵Labels


for k=1:K-1

    
      labeledSet = find(Labels ==k );
      Num(k)=length(labeledSet);
      NumMin=min(Num);

end

NumMin=50;
LabelSet=[];
for k=1:K-1

    [LabelSet1] =get_min_num_classes(Labels, k, NumMin);
    LabelSet=[LabelSet LabelSet1'];%得到5%的训练数据位置

end


Exam_Labels = Labels(LabelSet)'; %检查随机获取的位置信息的类标是否是每一类都是一样的个数，使类标的样本数目均衡
% % % trainData   = DATA(:, trainSet);% 获取数据并显示直方图

%% obtain Polar_Decompose features

path='D:\20200430Experiment\POLSAR Images\ESAR_Oberpfaffenhofen1200-1300\TouziWindows\17\T3\';
[DATA_Feat, DATA_Feat_order]=Touzi_Decomp_ALLPara_ESAR(path,r,c);

%% construct data matrix
DATA=DATA_Feat;

%% imshow the histogram of lambda
DATA1=DATA(1,:);
DATA2=DATA(2,:);
DATA3=DATA(3,:);
DATA4=DATA(4,:);

figure()
subplot(1,4,1),histogram(DATA1(:,LabelSet))
title('\lambda_1','FontSize',24)
xlabel('Bin','FontName','Times New Roman','FontSize',24)
ylabel('Numbers','FontName','Times New Roman','FontSize',24)

subplot(1,4,2),histogram(DATA2(:,LabelSet))
title('\lambda_2','FontSize',24)
xlabel('Bin','FontName','Times New Roman','FontSize',24)
ylabel('Numbers','FontName','Times New Roman','FontSize',24)
% ylim([0,5000]);



subplot(1,4,3),histogram(DATA3(:,LabelSet)) 
title('\lambda_3','FontSize',24)
xlabel('Bin','FontName','Times New Roman','FontSize',24)
ylabel('Numbers','FontName','Times New Roman','FontSize',24)
% ylim([0,11000]);

subplot(1,4,4),histogram(DATA4(:,LabelSet))
title('SPAN','FontSize',24)
xlabel('Bin','FontName','Times New Roman','FontSize',24)
ylabel('Numbers','FontName','Times New Roman','FontSize',24)

saveas(gcf,'Lambda_SPAN_ESAR.bmp')  % save three lambda with SPAN








% imshow three lambda 
figure()
subplot(1,3,1),histogram(DATA1(:,LabelSet))
title('\lambda_1')
xlabel('Bin','FontName','Times New Roman','FontSize',18)
ylabel('Numbers','FontName','Times New Roman','FontSize',18)

subplot(1,3,2),histogram(DATA2(:,LabelSet))
title('\lambda_2')
xlabel('Bin','FontName','Times New Roman','FontSize',18)
ylabel('Numbers','FontName','Times New Roman','FontSize',18)
% ylim([0,5000]);

subplot(1,3,3),histogram(DATA3(:,LabelSet)) 

title('\lambda_3')
xlabel('Bin','FontName','Times New Roman','FontSize',18)
ylabel('Numbers','FontName','Times New Roman','FontSize',18)
% ylim([0,11000]);

saveas(gcf,'Lambda_ESAR.bmp')
%% imshow the histogram of alpha
DATA5=DATA(5,:);
DATA6=DATA(6,:);
DATA7=DATA(7,:);

figure()
subplot(1,3,1),histogram(DATA5(:,LabelSet))
title('\alpha_{s1}')
xlabel('Bin','FontName','Times New Roman','FontSize',18)
ylabel('Numbers','FontName','Times New Roman','FontSize',18)

subplot(1,3,2),histogram(DATA6(:,LabelSet))
title('\alpha_{s2}')
xlabel('Bin','FontName','Times New Roman','FontSize',18)
ylabel('Numbers','FontName','Times New Roman','FontSize',18)

subplot(1,3,3),histogram(DATA7(:,LabelSet))
title('\alpha_{s3}')
xlabel('Bin','FontName','Times New Roman','FontSize',18)
ylabel('Numbers','FontName','Times New Roman','FontSize',18)

saveas(gcf,'Alpha_ESAR.bmp')
%% imshow the histogram of Phi
DATA8=DATA(8,:);
DATA9=DATA(9,:);
DATA10=DATA(10,:);

figure()
subplot(1,3,1),histogram(DATA8(:,LabelSet))
title('\Phi_{\alpha_{s1}}')
xlabel('Bin','FontName','Times New Roman','FontSize',18)
ylabel('Numbers','FontName','Times New Roman','FontSize',18)

subplot(1,3,2),histogram(DATA9(:,LabelSet))
title('\Phi_{\alpha_{s2}}')
xlabel('Bin','FontName','Times New Roman','FontSize',18)
ylabel('Numbers','FontName','Times New Roman','FontSize',18)

subplot(1,3,3),histogram(DATA10(:,LabelSet))
title('\Phi_{\alpha_{s3}}')
xlabel('Bin','FontName','Times New Roman','FontSize',18)
ylabel('Numbers','FontName','Times New Roman','FontSize',18)


saveas(gcf,'Phi_ESAR.bmp')

%% imshow the histogram of Psi


DATA11=DATA(11,:);
DATA12=DATA(12,:);
DATA13=DATA(13,:);


figure()
subplot(1,3,1),histogram(DATA11(:,LabelSet))
title('\Psi_1')
xlabel('Bin','FontName','Times New Roman','FontSize',18)
ylabel('Numbers','FontName','Times New Roman','FontSize',18)

subplot(1,3,2),histogram(DATA12(:,LabelSet))
title('\Psi_2')
xlabel('Bin','FontName','Times New Roman','FontSize',18)
ylabel('Numbers','FontName','Times New Roman','FontSize',18)

subplot(1,3,3),histogram(DATA13(:,LabelSet))
title('\Psi_3')
xlabel('Bin','FontName','Times New Roman','FontSize',18)
ylabel('Numbers','FontName','Times New Roman','FontSize',18)

saveas(gcf,'Psi_ESAR.bmp')
%% imshow the histogram of Tau
DATA14=DATA(14,:);
DATA15=DATA(15,:);
DATA16=DATA(16,:);

figure()
subplot(1,3,1),histogram(DATA14(:,LabelSet))
title('\tau_1')
xlabel('Bin','FontName','Times New Roman','FontSize',18)
ylabel('Numbers','FontName','Times New Roman','FontSize',18)

subplot(1,3,2),histogram(DATA15(:,LabelSet))
title('\tau_2')
xlabel('Bin','FontName','Times New Roman','FontSize',18)
ylabel('Numbers','FontName','Times New Roman','FontSize',18)

subplot(1,3,3),histogram(DATA16(:,LabelSet))
title('\tau_3')
xlabel('Bin','FontName','Times New Roman','FontSize',18)
ylabel('Numbers','FontName','Times New Roman','FontSize',18)

saveas(gcf,'tau_ESAR.bmp')

% % % % % % % %% 
% % % % % % % figure()
% % % % % % % image1=imread('E:\20200430Experiment\POLSAR Images\ESAR_Oberpfaffenhofen1200-1300\Alpha_ESAR.bmp');
% % % % % % % image2=imread('E:\20200430Experiment\POLSAR Images\ESAR_Oberpfaffenhofen1200-1300\Phi_ESAR.bmp');
% % % % % % % image3=imread('E:\20200430Experiment\POLSAR Images\ESAR_Oberpfaffenhofen1200-1300\Psi_ESAR.bmp');
% % % % % % % image4=imread('E:\20200430Experiment\POLSAR Images\ESAR_Oberpfaffenhofen1200-1300\tau_ESAR.bmp');
% % % % % % % subplot(2,2,1),imshow(image1,[])
% % % % % % % subplot(2,2,2),imshow(image2,[])
% % % % % % % subplot(2,2,3),imshow(image3,[])
% % % % % % % subplot(2,2,4),imshow(image4,[])
% % % % % % % 
% % % % % % % saveas(gcf,'Touzi_ESAR.bmp')

%%

% save ESAR_Balanced_Histogram_All_Paras.mat

