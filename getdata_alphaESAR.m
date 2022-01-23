function [trainSet , testSet] =getdata_alphaESAR(Labels,k,alpha) 

    labeledSet = find(Labels ==k );
    
    numTrain = round(numel(labeledSet)*alpha);
    
    bb=labeledSet(randperm(length(labeledSet)));
   
    trainSet = bb(1:numTrain);
%     testSet= bb(numTrain+1:end);
        testSet= labeledSet(1:end);
%       trainSet = labeledSet(1:numTrain); 
%       testSet= labeledSet(numTrain+1:end);
end