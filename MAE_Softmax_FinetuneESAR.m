                                        
function [pred1,pred2,acc1,acc2,pred_Un_BF,pred2_Un_AF,testSet,unlabeledSet,testLabels,...
   tr_ti1,tr_ti2,pr_ti1,pr_ti2]=MAE_Softmax_FinetuneESAR(inputSize,hiddenSizeL1,numClasses,...
                                            sparsityParam1,beta,lambda,softmaxLambda,...
                                            DATA,Labels,K,alpha)
%% �� ���ͼ�� �ҵ������λ��
unlabeledSet= find(Labels ==0 );
TotallabeledSet= find(Labels >=1 & Labels <=K-1 );

trainSet=[];
testSet=[];
for k=1:K-1
    [trainSet1 , testSet1] =getdata_alphaESAR(Labels,k,alpha);%�ֱ�õ�5%��ѵ������ʣ�����Ϊ��������
    trainSet=[trainSet trainSet1'];%�õ�5%��ѵ������λ��
    testSet=[testSet  testSet1']; %�õ�95%�Ĳ�������
end
%% δ��ǵ�����
unlabeledData = DATA(:, unlabeledSet);
%% ��ǵ������У�����ѵ�������ݺ��������Ե������Լ����Ǹ��Ե����
trainData   = DATA(:, trainSet);
trainLabels = Labels(trainSet)' ;

testData   = DATA(:, testSet);
testLabels = Labels(testSet)';   
%% ���ѡȡ5%��������Ϊ�ޱ�ǩѵ��������ѵ��SAE
a=unlabeledData;
num=size(a,2);
randidx = randperm(size(a,2)); %�������˳��
num_Sel =0.05*num; 
num_Sel =ceil(num_Sel);        %��Ҫ���ѡ�������� 
b = a(:,randidx(1:num_Sel));  %ѡ����%5���ޱ�ǩ����

%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to
%  change the parameters below.
% % % % % inputSize =9;              % Inputlayer 
% % % % % hiddenSizeL1= 30;          % Hidden Size of Layer 1 
% % % % % numClasses = 15;           % Outputlayer  Layer 3
% % % % % 
% % % % % sparsityParam1 = 0.05;     %  in the lecture notes).
% % % % % beta = 1;                 % weight of sparsity penalty term
% % % % % lambda = 1e-6;            % weight decay parameter

options.maxIter = 1000;    % Maximum number of iterations of L-BFGS to run
options.display = 'on';
%% STEP 1:Train the first sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled STL training images.
%  Randomly initialize the parameters
 tic
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

fprintf(2,'Begin to train the first sparse autoencoder\n');
addpath 'E:\20200430Experiment\minFunc\'
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                         % need a function pointer with two outputs: the
                         % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.

[sae1OptTheta, cost] = minFunc( @(p) mixturesparseAutoencoderCost(p,inputSize, hiddenSizeL1,lambda, sparsityParam1,beta,b), ...
                              sae1Theta, options);

%% STEP 2: Train the softmax classifier
 %  This trains the sparse autoencoder on the second autoencoder features.

 [sae1trainFeatures] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1,inputSize, trainData);
 
 sae1trainFeatures=[ones(1,size(sae1trainFeatures,2));sae1trainFeatures];%����һά�ؾ�+1
 %�������㲻��Ҫ��ʼ��softmaxģ�Ͳ���theta����Ϊ����softmaxTrain����������Զ���ʼ������������softmaxTrain�ġ�putinsize��������Ӧ�ð���
 %+1�Ľؾ���Ԫ�����������ǡ�putinsize����hiddenSizeL2+1
 
% parameters
% % % % softmaxLambda = 1e-8;
softoptions=struct;
softoptions.maxIter = 800;

fprintf(2,'Begin to train the softmax classifier\n');
softmaxModel = softmaxTrain(hiddenSizeL1+1, numClasses, softmaxLambda , sae1trainFeatures, trainLabels, softoptions);
 %ע��softmaxTrain������softmax_regression_vec������softmax_regression_vec���������y��Ҫ��һ������������Ϊ��sub2ind���������忴��  %���ڲ�ע�ͣ������������trainlabel
 %��Ҫ����������
 saeSoftmaxOptTheta = softmaxModel.optTheta(:);
 %softmaxģ�͵Ĳ����Ѿ�ѵ�����ˡ�saeSoftmaxOptTheta��һ����������
 tr_ti1=toc
%% STEP 3:Finetune softmax model
 % Initialize the stackedAETheta using the parameters learned
%���Ǵ�ǰ��Ĳ������Ѿ�˵��,����ѵ����ÿһ��ϡ���Ա��룬������Ҫ����ÿ��������ĵ�һ��Ĳ�������w��1����b��1����
 %stack{1}�ǵ�һ��ϡ���Ա��������w��1����b��1������,stack{2}��ǵ��ǵڶ���ϡ���Ա���w��1����b��1������
tic
stack = cell(1,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                    hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);

 % Initialize the parameters for the deep model
 [stackparams, netconfig] = stack2params(stack);%��ԭ��stack���һ��������ע�������������Ԫ�ش������ʲô��˳��Ҫ�졣
 stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];%�����������ģ�ͽ���΢��ʱ�����ĳ�ʼ�� 
 %stackedAETheta������ͷ(hiddenSizeL2 +1)* numClasses������ ��softmax ģ�͵Ĳ�����
% % % %  save('debug2') %��ǰ��������ĵõ������б���save һ�£���Ϊ����������Ҫ����д��wholeNetCost_and_grad���� �Ƿ���ȷ��
 %����Ϊԭģ��̫���ӣ����������˷�ʱ�䣬�������Ǽ���һ��ģ�ͣ����ģ�͵Ľṹ�Ͳ���ֵ�����ˡ������ټ������ȷ�Ժ�������ѵ�����΢����һ����
 %���ǻ�����Ҫԭ������ģ�͵�ֵ�����������ڼ�����������ȷ�Ժ���load һ��debug2.

 %%% ΢�� ѵ�� ǰ���stackedAETheta����������С�������еĳ�ʼ����������
% % % %  load debug2  %��ǰ��ı���ֵ�ټ���һ��
addpath 'E:\20200430Experiment\minFunc\'
 fprintf(2,'Begin to the three layers neural network\n');
 options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                           % function. Generally, for minFunc to work, you
                           % need a function pointer with two outputs: the
                           % function value and the gradient. In our problem,
                           % sparseAutoencoderCost.m satisfies this.
                                    
                                    
 [stackedAEOptTheta, cost] = minFunc( @(p) wholeNet2Cost_and_grad(p, ...
                                        inputSize, hiddenSizeL1,numClasses, ...
                                        lambda,softmaxLambda, sparsityParam1,beta,trainData,trainLabels), ...
                                        stackedAETheta, options);                                   
 %stackedAEOptTheta������������ģ�͵Ĳ�����
 tr_ti2=toc
%% STEP 4: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set

tic
[pred_Un_BF] = stackedAE2Predict(stackedAETheta, inputSize,hiddenSizeL1, numClasses,unlabeledData);
[pred1] = stackedAE2Predict(stackedAETheta, inputSize,hiddenSizeL1, numClasses, testData);
pr_ti1=toc

acc1 = mean(testLabels(:) == pred1(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc1 * 100);

%
tic
[pred2_Un_AF] = stackedAE2Predict(stackedAEOptTheta, inputSize,hiddenSizeL1, numClasses,unlabeledData);
[pred2] = stackedAE2Predict(stackedAEOptTheta, inputSize,hiddenSizeL1, numClasses,testData);
pr_ti2=toc

acc2 = mean(testLabels(:) == pred2(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc2 * 100);

toc
end
