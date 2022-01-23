function [pred_Te,acc_Te,pred_Un,testSet,unlabeledSet,testLabels,tr_Time,pr_Time]=Softmax_Classifier_JustinESAR(inputSize,numClasses,softmaxLambda,DATA,Labels,K,alpha)
% % % %% Paras.
% % % inputSize=Dim;
% % % numClasses=15;
% % % softmaxLambda=1e-6;

%% �����(Labels)ͼ�� �ҵ������λ��
unlabeledSet= find(Labels ==0 );
TotallabeledSet= find(Labels >=1 & Labels <=K-1 );

trainSet=[];
testSet=[];
for k=1:K-1
    [trainSet1 , testSet1] =getdata_alphaESAR(Labels,k,alpha);%�ֱ�õ�5%��ѵ������ʣ�����Ϊ��������
    trainSet=[trainSet trainSet1'];%�õ�5%��ѵ������λ��
    testSet=[testSet  testSet1']; %�õ�95%�Ĳ�������
end
%% data preprocessing
[Dim,~]=size(DATA);
% DATA=ZCA_Whitening(DATA);
%% δ��ǵ�����
unlabeledData = DATA(:, unlabeledSet);
%% ��ǵ������У�����ѵ�������ݺ��������Ե������Լ����Ǹ��Ե����
trainData   = DATA(:, trainSet);
trainLabels = Labels(trainSet)';
testData   = DATA(:, testSet);
testLabels = Labels(testSet)';

%% training.......
tic
[softmaxModel] = softmaxTrain(inputSize, numClasses, softmaxLambda, trainData, trainLabels);
tr_Time=toc
%% testing.....
tic
[pred_Te] = softmaxPredict(softmaxModel, testData);
[pred_Un] = softmaxPredict(softmaxModel, unlabeledData );
pr_Time=toc

acc_Te = mean(testLabels(:) == pred_Te(:));
fprintf('Test Accuracy with Softmax classifier and Features with ZCAwhitening: %0.3f%%\n', acc_Te * 100);

end

