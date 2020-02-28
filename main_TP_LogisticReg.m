%------------------------------------
%   LOGISTIC REGRESSION CLASSIFIER
%------------------------------------

clear
clc
close all

%% STEP 1: DESCRIPTION

% Load the images and labels
load Horizontal_edges;

% Number of images of cutting edges
num_edges = length(horiz_edges);

% Number of features (in the case of ShapeFeat it is 10)
num_features = 10;

% Initialiation of matrix of descriptors. It will have a size (m x n), where
% m is the number of training patterns (i.e. elements) and n is the number 
% of features (i.e. the length of the feature vector which characterizes 
% the cutting edge).
X = zeros(num_edges, num_features);

% Describe the images of the horizontal edges by calling the fGetShapeFeat 
% function
for i=1:num_edges
    
    % Get the i-th cutting edge
    edge = logical(horiz_edges{i}); % DON'T REMOVE
    
    % Compute the descriptors of the cutting edge usign the fGetShapeFeat
    % function
    desc_edge_i = fGetShapeFeat(edge);
    
    % Store the feature vector into the matrix X.
    X(i,:) = desc_edge_i;
end

% Create the vector of labels Y. Y(j) will store the label of the curring
% edge represented by the feature vector contained in the j-th row of the 
% matrix X.
% The problem will be binary: class 0 correspond to a low or medium wear
% level, whereas class 1 correspond to a high wear level.
Y = labels(:,2)'>=2;

save('tool_descriptors.mat', 'X', 'Y');

%% STEP 2: CLASSIFICATION

%% PRELIMINARY: LOAD DATASET AND PARTITION TRAIN-TEST SETS (NO NEED TO CHANGE ANYTHING)

clear
clc
close all

load tool_descriptors;
% X contains the training patterns (dimension 10)
% Y contains the class label of the patterns (i.e. Y(37) contains the label
% of the pattern X(37,:) ).

% Number of patterns (i.e., elements) and variables per pattern in this
% dataset


% Normalization of the data
X=normalize(X);
[num_patterns, num_features] = size(X);
% mu_data = mean(X);
% std_data = std(X);
% X = (X-repmat(mu_data,length(X),1))./(repmat(std_data,length(X),1));

% Parameter that indicates the percentage of patterns that will be used for
% the training
p_train = 0.8;

% SPLIT DATA INTO TRAINING AND TEST SETS

num_patterns_train = round(p_train*num_patterns);

indx_permutation = randperm(num_patterns);

indxs_train = indx_permutation(1:num_patterns_train);
indxs_test = indx_permutation(num_patterns_train+1:end);

X_train = X(indxs_train, :);
Y_train = Y(indxs_train);

X_test= X(indxs_test, :);
Y_test = Y(indxs_test);


%% PART 2.1: TRAINING OF THE CLASSIFIER AND CLASSIFICATION OF THE TEST SET

% Learning rate. Change it accordingly, depending on how the cost function
% evolve along the iterations


alpha = 2;

max_iter=1000;

% The function fTrain_LogisticReg implements the logistic regression 
% classifier. Open it and complete the code.

% TRAINING
theta = fTrain_LogisticReg(X_train, Y_train, alpha,max_iter);

% CLASSIFICATION OF THE TEST SET
Y_test_hat = fClassify_LogisticReg(X_test, theta);
% 
% % Assignation of the class
Y_test_asig = Y_test_hat>=0.5;
% SVModel=fitcsvm(X_train,Y_train);
% Y_test_asig = predict(SVModel,X_test);


% 
% %% PART 2.2: PERFORMANCE OF THE CLASSIFIER: CALCULATION OF THE ACCURACY AND FSCORE
% 
% % % Show confusion matrix
figure;
plotconfusion(Y_test, Y_test_asig);


% % % ACCURACY AND F-SCORE
% % % ====================== YOUR CODE HERE ======================
n=length(Y_test);
TP=0;
TN=0;
FP=0;
FN=0;
for i=1:n
    if(Y_test(i)==1 && Y_test_asig(i)==1)
        TP=TP+1;
    end
    
     if(Y_test(i)==0 && Y_test_asig(i)==0)
        TN=TN+1;
     end
     
     if(Y_test(i)==0 && Y_test_asig(i)==1)
        FP=FP+1;
     end
     
     if(Y_test(i)==1 && Y_test_asig(i)==0)
        FN=FN+1;
    end
end
accuracy=(TP+TN)/n;
precision=TP/(TP+FP);
recall=TP/(TP+FN);
FScore=2*precision*recall/(precision + recall);
acc(kk,1)=accuracy;
acc(kk,2)=FScore;


% % % % ============================================================
% % % 
% 
% fprintf('\n******\nAccuracy = %1.4f%% (classification)\n', accuracy*100);
% fprintf('\n******\nFScore = %1.4f (classification)\n', FScore);
% 
% %%% ===================== THE ROC  CURVE and models' comparison =========================
% [tpr,fpr,thresholds] = roc(Y_test,Y_test_hat);
% figure(3)
% %plot(fpr,tpr,[0,1],[0,1],'--');
% pred=X_train;
% resp=Y_train;
% mdl = fitglm(pred,resp,'Distribution','binomial','Link','logit');
% score_log = mdl.Fitted.Probability;
% [Xlog,Ylog,Tlog,AUClog] = perfcurve(resp,score_log,'true');
% SVMmodel=fitcsvm(X_train,Y_train);
% NBmodel=fitcnb(X_train,Y_train);
% [~,score_nb] = resubPredict(NBmodel);
% [~,score_svm] = resubPredict(SVMmodel);
% [Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(resp,score_svm(:,SVMmodel.ClassNames),'true');
% [Xnb,Ynb,Tnb,AUCnb] = perfcurve(resp,score_nb(:,NBmodel.ClassNames),'true');
% 
% plot(Xlog,Ylog)
% hold on
% plot(Xsvm,Ysvm)
% plot(Xnb,Ynb)
% legend('Logistic Regression','Support Vector Machines','Naive Bayes','Location','Best')
% xlabel('False positive rate'); ylabel('True positive rate');
% title('ROC Curves for Logistic Regression, SVM, and Naive Bayes Classification')
% hold off
% 
% %%===========The area under the cureve================
% AUClog
% AUCsvm
% AUCnb
% %%---------------- Principale components analysis--------
% pcacoef=pca(X);
% pcap=pcacoef(:,1:2);
% X=X*pcap;
% 
% [num_patterns, num_features] = size(X);
% p_train = 0.8;
% 
% % SPLIT DATA INTO TRAINING AND TEST SETS
% 
% num_patterns_train = round(p_train*num_patterns);
% 
% indx_permutation = randperm(num_patterns);
% 
% indxs_train = indx_permutation(1:num_patterns_train);
% indxs_test = indx_permutation(num_patterns_train+1:end);
% 
% X_train = X(indxs_train, :);
% Y_train = Y(indxs_train);
% 
% X_test= X(indxs_test, :);
% Y_test = Y(indxs_test);
% 
% pred=X_train;
% resp=Y_train;
% SVMmodel=fitcsvm(X_train,Y_train);
% err = loss(SVMmodel,X,Y);
% 
% accuracy_pca=1-err