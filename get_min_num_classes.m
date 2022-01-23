function [LabelSet] =get_min_num_classes(Labels,k,minVal)

    labeledSet = find(Labels ==k );
    bb=labeledSet(randperm(length(labeledSet)));
    LabelSet = bb(1:minVal);

end