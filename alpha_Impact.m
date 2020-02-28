
%clear
% load tool_descriptors;
 ALPHA=[0.01,0.1,0.5,1,2,3,10,15,20];
% acc=zeros(length(ALPHA),2);
% k=1;
% max_iter=1000;
% for alpha=ALPHA
%     [acc(k,1),acc(k,2)]=cross_val(X,Y,alpha,max_iter);
%     k=k+1;
% end
load metrics
figure

p=plot(ALPHA,acc(:,1),ALPHA,acc(:,2));

p(1).LineWidth = 2;
p(2).LineWidth = 2;

p(1).Marker = '*';
p(2).Marker = '*';

legend('accuracy','F1-score');
xlabel('alpha'); ylabel('performance metric');

title('the impact of the hyper parameter alpha on the performance of the model')

