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

%% 给出应用的各类数据的特征信息，构成数据矩阵
%% obtain Polar_Decompose features
path='E:\20200430Experiment\POLSAR Images\ESAR_Oberpfaffenhofen1200-1300\TouziWindows\17\T3\';
[DATA_Feat, DATA_Feat_order]=Touzi_Decomp_ALLPara_ESAR(path,r,c);


%% construct data matrix
DATA=DATA_Feat;


DATA_part=[DATA_Feat_order(1,:);DATA_Feat_order(3:4,:);DATA_Feat_order(5:9,:);DATA_Feat_order(11:12,:)];

%% Paras.
[Dim,~]=size(DATA);
inputSize =Dim;              % Inputlayer 
hiddenSizeL1= 30;          % Hidden Size of Layer 1 
hiddenSizeL2 = 60;         % Hidden Size of Layer 2
numClasses = K-1;           % Outputlayer  Layer 3

sparsityParam1 = 0.05;     %  in the lecture notes).
sparsityParam2 = 0.05;     %  in the lecture notes).
beta = 1;                  % weight of sparsity penalty term
lambda = 1e-6;             % weight decay parameter
softmaxLambda = 1e-6;

Ntimes=30;
alpha=0.01;
%% demo function2
addpath 'E:\20200430Experiment\data_preprocessing\'

DATA2=DATA_part;
DATA2=ZCA_Whitening(DATA2);
[Dim2,~]=size(DATA2);


for kk=1:Ntimes
    
    [pred1_SM_Te(kk,:),acc1_SM_Te(kk),pred1_SM_Un(kk,:),testSet_SM(kk,:),unlabeledSet_SM(:,kk),...
                 testLabels_SM(kk,:),tr_Time_SM(kk),pr_Time_SM(kk)]=Softmax_Classifier_JustinESAR(...
                                             Dim2,numClasses,softmaxLambda,DATA2,Labels,K,alpha);
    
    [pred2_Te_BF(kk,:),pred3_Te_AF(kk,:),acc2_Te_BF(kk),acc3_Te_AF(kk),pred2_Un_BF(kk,:),pred3_Un_AF(kk,:),...
               testSet_AE(kk,:),unlabeledSet_AE(:,kk),testLabels_AE(kk,:),...
               tr2(kk),tr3(kk),pr2(kk),pr3(kk)]=AE_Softmax_FinetuneESAR(Dim2,hiddenSizeL1,numClasses,...
                       sparsityParam1,beta,lambda,softmaxLambda,...
                       DATA2,Labels,K,alpha);
end

%% Randomly select a result as the display result in the paper
addpath 'E:\20200430Experiment\Generator_Visualization_Image\'
bbb=randperm(Ntimes);
SS=bbb(1);

%% imshow the result of  Softmax + Partial Features
% imshow the prediction on the groundtruth data
WWimage_TEST_SM = Genereator_Predict_TEST(unlabeledSet_SM(:,SS) ,testSet_SM(SS,:),pred1_SM_Te(SS,:),indexing,r,c);
imwrite(uint8(WWimage_TEST_SM ),'./GT_ESARimage_SM.bmp');  
% imshow the prediction on the All Pixels 
WWimage_ALL_SM = Genereator_Predict_All(unlabeledSet_SM(:,SS),testSet_SM(SS,:),pred1_SM_Un(SS,:),pred1_SM_Te(SS,:),indexing,r,c);
imwrite(uint8(WWimage_ALL_SM),'./AP_ESARimage_SM.bmp');                           
% % % % % % % % % % % % % % % % % % % % % % %

% get the mean and standard of Ntimes results
for kk=1:Ntimes
    [acc1_SM_ALL(kk), recall1_SM_ALL(kk,:)] =accandrecall(testLabels_SM(kk,:),pred1_SM_Te(kk,:));
end
accMea_SM=mean(acc1_SM_ALL);
accStd_SM=std(acc1_SM_ALL);

recallMea_SM=mean (recall1_SM_ALL);
recallStd_SM=std( recall1_SM_ALL,0,1);

hor_header={'Softmax classifier','mean_std'};
ver_header={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,'OA','train time','test time'};
Mean_col_SM=[recallMea_SM,accMea_SM,mean(tr_Time_SM),mean(pr_Time_SM)]';
Std_col_SM=[recallStd_SM,accStd_SM,std(tr_Time_SM),std(pr_Time_SM)]';
Mea_Std_SM=[Mean_col_SM,Std_col_SM];

xlswrite('SM.xls',hor_header,1,'B1:C1')
xlswrite('SM.xls',ver_header',1,'A2:A19')
xlswrite('SM.xls',Mea_Std_SM,1,'B2')   

%
[accuracy1] = Genereator_Predict_TEST_Quantitative_index(pred1_SM_Te(SS,:),indexing,K,testLabels_SM(SS,:));

%% imshow the result of AE + Partial Features + Softmax classifier
% imshow the prediction on the groundtruth data
WWimage_TEST2_AE= Genereator_Predict_TEST(unlabeledSet_AE(:,SS) ,testSet_AE(SS,:),pred2_Te_BF(SS,:),indexing,r,c);
imwrite(uint8(WWimage_TEST2_AE ),'./GT_ESARimage_AE.bmp');  
% imshow the prediction on the All Pixels 
WWimage_ALL2_AE = Genereator_Predict_All(unlabeledSet_AE(:,SS),testSet_AE(SS,:),pred2_Un_BF(SS,:),pred2_Te_BF(SS,:),indexing,r,c);
imwrite(uint8(WWimage_ALL2_AE),'./AP_ESARimage_AE.bmp');                           
% % % % % % % % % % % % % % % % % % % % % % % % % %

% get the mean and standard of Ntimes results
for kk=1:Ntimes
    [acc2_AE_ALL(kk), recall2_AE_ALL(kk,:)] =accandrecall(testLabels_AE(kk,:),pred2_Te_BF(kk,:));
end
accMea_AE=mean(acc2_AE_ALL);
accStd_AE=std(acc2_AE_ALL);

recallMea_AE=mean (recall2_AE_ALL);
recallStd_AE=std( recall2_AE_ALL,0,1);

Mean_col_AE=[recallMea_AE,accMea_AE,mean(tr2),mean(pr2)]';
Std_col_AE=[recallStd_AE,accStd_AE,std(tr2),std(pr2)]';
Mea_Std_AE=[Mean_col_AE,Std_col_AE];
hor_header={'AE','mean_std'};
ver_header={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,'OA','train time','test time'};
xlswrite('AE.xls',hor_header,1,'B1:C1')
xlswrite('AE.xls',ver_header',1,'A2:A19')
xlswrite('AE.xls',Mea_Std_AE,1,'B2')   

%
[accuracy2] = Genereator_Predict_TEST_Quantitative_index(pred2_Te_BF(SS,:),indexing,K,testLabels_AE(SS,:));


%% imshow the result of AE + Softmax + Finetune + Partial Features
% imshow the prediction on the groundtruth data
WWimage_TEST3_AEFT = Genereator_Predict_TEST(unlabeledSet_AE(:,SS) ,testSet_AE(SS,:),pred3_Te_AF(SS,:),indexing,r,c);
imwrite(uint8(WWimage_TEST3_AEFT ),'./GT_ESARimage_AE_FT.bmp');  
% imshow the prediction on the All Pixels 
WWimage_ALL3_AEFT = Genereator_Predict_All(unlabeledSet_AE(:,SS),testSet_AE(SS,:),pred3_Un_AF(SS,:),pred3_Te_AF(SS,:),indexing,r,c);
imwrite(uint8(WWimage_ALL3_AEFT),'./AP_ESARimage_AE_FT.bmp');                           
% % % % % % % % % % % % % % % % % % %


% get the mean and standard of Ntimes results
for kk=1:Ntimes
    [acc3_AEFT_ALL(kk), recall3_AEFT_ALL(kk,:)] =accandrecall(testLabels_AE(kk,:),pred3_Te_AF(kk,:));
end
accMea_AEFT=mean(acc3_AEFT_ALL);
accStd_AEFT=std(acc3_AEFT_ALL);

recallMea_AEFT=mean (recall3_AEFT_ALL);
recallStd_AEFT=std( recall3_AEFT_ALL,0,1);

Mean_col_AEFT=[recallMea_AEFT,accMea_AEFT,mean(tr3),mean(pr3)]';
Std_col_AEFT=[recallStd_AEFT,accStd_AEFT,std(tr3),std(pr3)]';
Mea_Std_AEFT=[Mean_col_AEFT,Std_col_AEFT];
hor_header={'AEFT','mean_std'};
ver_header={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,'OA','train time','test time'};
xlswrite('AEFT.xls',hor_header,1,'B1:C1')
xlswrite('AEFT.xls',ver_header',1,'A2:A19')
xlswrite('AEFT.xls',Mea_Std_AEFT,1,'B2')   

%
[accuracy3] = Genereator_Predict_TEST_Quantitative_index(pred3_Te_AF(SS,:),indexing,K,testLabels_AE(SS,:));


%% 根据图像中数据的分布直方图，大致认为服从的分布进行分别预处理
addpath 'E:\20200430Experiment\data_preprocessing\'

%
Data1=DATA_part(1:3,:);
Data_1=data_normalized(Data1);

%
Data2=DATA_part(4:10,:);
Data_2=ZCA_Whitening(Data2);

% % % % % % % % % % %
% % % % % % % % % % Data3=DATA_part(12:13,:);
% % % % % % % % % % AA=std(Data3,0,2);
% % % % % % % % % % MM=mean(Data3,2);
% % % % % % % % % % AAA=repmat(AA,1,r*c);
% % % % % % % % % % MMM=repmat(MM,1,r*c);
% % % % % % % % % % Data_3 =(Data3-MMM)./AAA;
%%
DATA1=[Data_1;Data_2];
[Dim1,~]=size(DATA1);

addpath 'E:\20200430Experiment\'
for kk=1:Ntimes
% % %     [pred5ZP(kkk,:),acc5ZP(kkk)]=Softmax_Classifier_Justin(Dim1,numClasses,softmaxLambda,DATA1,Labels,K);
    
% % %     [pred6ZP(kkk,:),pred7ZP(kkk,:),acc6ZP(kkk),acc7ZP(kkk)]=AE_Softmax_Finetune(Dim1,hiddenSizeL1,numClasses,...
% % %         sparsityParam1,beta,lambda,softmaxLambda,...
% % %         DATA2,Labels,K);



 [pred4_Te_BF(kk,:),pred5_Te_AF(kk,:),acc4_Te_BF(kk),acc5_Te_AF(kk),pred4_Un_BF(kk,:),pred5_Un_AF(kk,:),...
     testSet_MAE(kk,:),unlabeledSet_MAE(:,kk),testLabels_MAE(kk,:),...
   tr_ti4(kk),tr_ti5(kk),pr_ti4(kk),pr_ti5(kk)]=MAE_Softmax_FinetuneESAR(Dim1,hiddenSizeL1,numClasses,...
                                            sparsityParam1,beta,lambda,softmaxLambda,...
                                            DATA1,Labels,K,alpha);
    
% % % % %     [pred8ZP(kk,:),pred9ZP(kk,:),acc8ZP(kk),acc9ZP(kk)]=MAE_Softmax_Finetune(Dim1,hiddenSizeL1,numClasses,...
% % % % %         sparsityParam1,beta,lambda,softmaxLambda,...
% % % % %         DATA1,Labels,K);
end

%% imshow the result of AE + Partial Features +Softmax classifier 
% imshow the prediction on the groundtruth data
WWimage_TEST4_MAE = Genereator_Predict_TEST(unlabeledSet_MAE(:,SS) ,testSet_MAE(SS,:),pred4_Te_BF(SS,:),indexing,r,c);
imwrite(uint8(WWimage_TEST4_MAE ),'./GT_ESARimage_MAE.bmp');  
% imshow the prediction on the All Pixels 
WWimage_ALL4_MAE = Genereator_Predict_All(unlabeledSet_MAE(:,SS),testSet_MAE(SS,:),pred4_Un_BF(SS,:),pred4_Te_BF(SS,:),indexing,r,c);
imwrite(uint8(WWimage_ALL4_MAE),'./AP_ESARimage_MAE.bmp');                           

% get the mean and standard of Ntimes results
for kk=1:Ntimes
    [acc4_MAE_ALL(kk), recall4_MAE_ALL(kk,:)] =accandrecall(testLabels_MAE(kk,:),pred4_Te_BF(kk,:));
end
accMea_MAE=mean(acc4_MAE_ALL);
accStd_MAE=std(acc4_MAE_ALL);

recallMea_MAE=mean (recall4_MAE_ALL);
recallStd_MAE=std( recall4_MAE_ALL,0,1);

Mean_col_MAE=[recallMea_MAE,accMea_MAE,mean(tr_ti4),mean(pr_ti4)]';
Std_col_MAE=[recallStd_MAE,accStd_MAE,std(tr_ti4),std(pr_ti4)]';
Mea_Std_MAE=[Mean_col_MAE,Std_col_MAE];
hor_header={'MAE','mean_std'};
ver_header={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,'OA','train time','test time'};
xlswrite('MAE.xls',hor_header,1,'B1:C1')
xlswrite('MAE.xls',ver_header',1,'A2:A19')
xlswrite('MAE.xls',Mea_Std_MAE,1,'B2')   

%
[accuracy4] = Genereator_Predict_TEST_Quantitative_index(pred4_Te_BF(SS,:),indexing,K,testLabels_MAE(SS,:));

%% imshow the result of AE + Softmax + Finetune + Partial Features
% imshow the prediction on the groundtruth data
WWimage_TEST5_MAEFT = Genereator_Predict_TEST(unlabeledSet_MAE(:,SS) ,testSet_MAE(SS,:),pred5_Te_AF(SS,:),indexing,r,c);
imwrite(uint8(WWimage_TEST5_MAEFT ),'./GT_ESARimage_MAE_FT.bmp');  
% imshow the prediction on the All Pixels 
WWimage_ALL5_MAEFT = Genereator_Predict_All(unlabeledSet_MAE(:,SS),testSet_MAE(SS,:),pred5_Un_AF(SS,:),pred5_Te_AF(SS,:),indexing,r,c);
imwrite(uint8(WWimage_ALL5_MAEFT),'./AP_ESARimage_MAE_FT.bmp');                           

% get the mean and standard of Ntimes results
for kk=1:Ntimes
    [acc5_MAEFT_ALL(kk), recall5_MAEFT_ALL(kk,:)] =accandrecall(testLabels_MAE(kk,:),pred5_Te_AF(kk,:));
end
accMea_MAEFT=mean(acc5_MAEFT_ALL);
accStd_MAEFT=std(acc5_MAEFT_ALL);

recallMea_MAEFT=mean (recall5_MAEFT_ALL);
recallStd_MAEFT=std( recall5_MAEFT_ALL,0,1);

Mean_col_MAEFT=[recallMea_MAEFT,accMea_MAEFT,mean(tr_ti5),mean(pr_ti5)]';
Std_col_MAEFT=[recallStd_MAEFT,accStd_MAEFT,std(tr_ti5),std(pr_ti5)]';
Mea_Std_MAEFT=[Mean_col_MAEFT,Std_col_MAEFT];
hor_header={'MAEFT','mean_std'};
ver_header={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,'OA','train time','test time'};
xlswrite('MAEFT.xls',hor_header,1,'B1:C1')
xlswrite('MAEFT.xls',ver_header',1,'A2:A19')
xlswrite('MAEFT.xls',Mea_Std_MAEFT,1,'B2')   
%
[accuracy5] = Genereator_Predict_TEST_Quantitative_index(pred5_Te_AF(SS,:),indexing,K,testLabels_MAE(SS,:));


%%

% % % % % % % % % % % %
% % save ESAR_Ntimes_5methods.mat
