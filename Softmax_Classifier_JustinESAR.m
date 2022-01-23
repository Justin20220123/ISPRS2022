function [pred_Te,acc_Te,pred_Un,testSet,unlabeledSet,testLabels,tr_Time,pr_Time]=Softmax_Classifier_JustinESAR(inputSize,numClasses,softmaxLambda,DATA,Labels,K,alpha)
% % % %% Paras.
% % % inputSize=Dim;
% % % numClasses=15;
% % % softmaxLambda=1e-6;

%% 从类标(Labels)图中 找到各类的位置
unlabeledSet= find(Labels ==0 );
TotallabeledSet= find(Labels >=1 & Labels <=K-1 );

trainSet=[];
testSet=[];
for k=1:K-1
    [trainSet1 , testSet1] =getdata_alphaESAR(Labels,k,alpha);%分别得到5%的训练数据剩余的作为测试数据
    trainSet=[trainSet trainSet1'];%得到5%的训练数据位置
    testSet=[testSet  testSet1']; %得到95%的测试数据
end
%% data preprocessing
[Dim,~]=size(DATA);
% DATA=ZCA_Whitening(DATA);
%% 未标记的数据
unlabeledData = DATA(:, unlabeledSet);
%% 标记的数据中，用来训练的数据和用来测试的数据以及它们各自的类标
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

