function [acc,Fscore]=cross_val(X,Y,alpha,max_iter)

n=length(X);
TP=0;
TN=0;
FP=0;
FN=0;
for i=1:n
    index=ones(1,n);
    index(i)=0;
    index=logical(index);
    Xtrain=X(index,:);
    Ytrain=Y(index);
    Xtest=X(i,:);
    Y_test=Y(i);
    theta = fTrain_LogisticReg(Xtrain, Ytrain, alpha,max_iter);
    Y_test_hat = fClassify_LogisticReg(Xtest, theta);
    Y_test_asig = Y_test_hat>=0.5;

    
    if(Y_test==1 && Y_test_asig==1)
        TP=TP+1;
    end
    
     if(Y_test==0 && Y_test_asig==0)
        TN=TN+1;
     end
     
     if(Y_test==0 && Y_test_asig==1)
        FP=FP+1;
     end
     
     if(Y_test==1 && Y_test_asig==0)
        FN=FN+1;
    end
end
acc=(TP+TN)/n;
precision=TP/(TP+FP);
recall=TP/(TP+FN);
Fscore=2*precision*recall/(precision + recall);
end