function [P] = SFS(cc,X,YY,YYl,Nl,param,NITER)
% Semi-supervised Feature Selection with Binary Label Learning (SFS-BML)

% X:dd*N; X=[Xl;Xu]; dd:the dimension of features; N:the number of samples;
% YY:N*cc, labels of data, YY=[YYl;YYu];
% YYl: labels of labelled samples, YYu=zeros(Nl,cc);
% Nl: the number of labelled samples;

% X = NormalizeFea(X,0);
% X = mapminmax(X,0,1);
[dd,N] = size(X);
options = [];
options.Metric = 'HeatKernel';
options.NeighborMode = 'KNN';
options.WeightMode = 'Cosine';
options.k = 10;
A0 = constructW(X',options);
A0 = A0-diag(diag(A0));
A = (A0+A0')/2;
D = diag(sum(A));
L = D - A;
%% Initialization
Wl0 = ones(Nl,1);
Wu0 = ones(N-Nl,1);
W0 = [Wl0;Wu0];
W = diag(W0);
ZZ1 = param.alpha*ones(Nl,1);
Z1 = diag(ZZ1);
ZZ2 = zeros(N-Nl,1);
Z2 = diag(ZZ2);
Z = blkdiag(Z1,Z2);

P = zeros(dd,cc);
B = randn(N,cc)>0;
B = B*1;
F = B;
G = B-F;
%% Optimization
for iter = 1:NITER
    % update P
    Pi = sqrt(sum(P.*P,2)+eps);
    hh = 0.5./Pi;
    H = diag(hh);
    P = inv(X*W*X'+param.lambda*H)*X*W*B;
    % update G
    G = G+param.rhoo*(B-F); 
    % update W
    Wi = zeros(1,N);
    temp3=0;
    for i = 1:N
        temp1(:,i) = (P'*X(:,i)-B(i,:)').^2;
        temp2(1,i) = sqrt(sum(temp1(:,i)));
        temp3 = temp3+temp2(1,i);
    end
    for i = 1:N
        Wi(1,i) = 1/(temp2(1,i)/temp3);
    end
    W = diag(Wi);
    % update B
    for i = 1:Nl
        B(i,:) = ((sign(param.alpha*YYl(i,:)+param.rhoo*F(i,:)-G(i,:)))+1)/2;
    end
    for j = Nl+1:N
        B(j,:) = ((sign(F(j,:)-G(j,:)/param.rhoo))+1)/2;
    end
    % update F
    I1 = eye(N);
    tempf1 = inv(2*L+param.rhoo*I1+Z+2*param.beta*W);
    tempf2 = param.rhoo*B+G+Z*YY+2*param.beta*W*X'*P;
    F = tempf1*tempf2;
end
end

