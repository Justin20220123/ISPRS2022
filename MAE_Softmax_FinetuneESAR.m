                                        
function [pred1,pred2,acc1,acc2,pred_Un_BF,pred2_Un_AF,testSet,unlabeledSet,testLabels,...
   tr_ti1,tr_ti2,pr_ti1,pr_ti2]=MAE_Softmax_FinetuneESAR(inputSize,hiddenSizeL1,numClasses,...
                                            sparsityParam1,beta,lambda,softmaxLambda,...
                                            DATA,Labels,K,alpha)
%% 从 类标图中 找到各类的位置
unlabeledSet= find(Labels ==0 );
TotallabeledSet= find(Labels >=1 & Labels <=K-1 );

trainSet=[];
testSet=[];
for k=1:K-1
    [trainSet1 , testSet1] =getdata_alphaESAR(Labels,k,alpha);%分别得到5%的训练数据剩余的作为测试数据
    trainSet=[trainSet trainSet1'];%得到5%的训练数据位置
    testSet=[testSet  testSet1']; %得到95%的测试数据
end
%% 未标记的数据
unlabeledData = DATA(:, unlabeledSet);
%% 标记的数据中，用来训练的数据和用来测试的数据以及它们各自的类标
trainData   = DATA(:, trainSet);
trainLabels = Labels(trainSet)' ;

testData   = DATA(:, testSet);
testLabels = Labels(testSet)';   
%% 随机选取5%的数据作为无标签训练数据来训练SAE
a=unlabeledData;
num=size(a,2);
randidx = randperm(size(a,2)); %随机打乱顺序
num_Sel =0.05*num; 
num_Sel =ceil(num_Sel);        %需要随机选出的列数 
b = a(:,randidx(1:num_Sel));  %选出的%5的无标签数据

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
 
 sae1trainFeatures=[ones(1,size(sae1trainFeatures,2));sae1trainFeatures];%增加一维截距+1
 %在这里你不需要初始化softmax模型参数theta，因为下面softmaxTrain函数里面会自动初始化，还有输入softmaxTrain的‘putinsize’和样本应该包括
 %+1的截距神经元，所以这里是‘putinsize’是hiddenSizeL2+1
 
% parameters
% % % % softmaxLambda = 1e-8;
softoptions=struct;
softoptions.maxIter = 800;

fprintf(2,'Begin to train the softmax classifier\n');
softmaxModel = softmaxTrain(hiddenSizeL1+1, numClasses, softmaxLambda , sae1trainFeatures, trainLabels, softoptions);
 %注意softmaxTrain里面有softmax_regression_vec函数，softmax_regression_vec函数输入的y需要是一个列向量（因为有sub2ind函数，具体看函  %数内部注释），所以这里的trainlabel
 %需要是列向量。
 saeSoftmaxOptTheta = softmaxModel.optTheta(:);
 %softmax模型的参数已经训练好了。saeSoftmaxOptTheta是一个列向量。
 tr_ti1=toc
%% STEP 3:Finetune softmax model
 % Initialize the stackedAETheta using the parameters learned
%我们从前面的博客中已经说了,我们训练好每一个稀疏自编码，我们需要的是每个神经网络的第一层的参数，即w（1）和b（1），
 %stack{1}是第一个稀疏自编码里面的w（1）和b（1）参数,stack{2}标记的是第二个稀疏自编码w（1）和b（1）参数
tic
stack = cell(1,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                    hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);

 % Initialize the parameters for the deep model
 [stackparams, netconfig] = stack2params(stack);%把原胞stack变成一个向量，注意这个向量里面元素代表的是什么，顺序不要混。
 stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];%这个就是整个模型进行微调时参数的初始化 
 %stackedAETheta向量的头(hiddenSizeL2 +1)* numClasses个参数 是softmax 模型的参数。
% % % %  save('debug2') %把前面运行完的得到的所有变量save 一下，因为接下来我们要看看写得wholeNetCost_and_grad函数 是否正确，
 %又因为原模型太复杂，运行起来浪费时间，所以我们简化了一下模型，因此模型的结构和参数值都变了。我们再检测完正确以后，我们想训练这个微调这一步。
 %我们还是需要原来整个模型的值，所以我们在检测这个函数正确以后，再load 一下debug2.

 %%% 微调 训练 前面的stackedAETheta就是这里最小化过程中的初始化参数向量
% % % %  load debug2  %把前面的变量值再加载一遍
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
 %stackedAEOptTheta就是最终整个模型的参数。
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
