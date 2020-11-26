function output(partId)
% Random Test Cases
X = [ones(20,1) (exp(1) * sin(1:1:20))' (exp(0.5) * cos(1:1:20))'];
y = sin(X(:,1) + X(:,2)) > 0;

if partId == '0'
    plotData(X, y);
    
elseif partId == '1'
    disp(sigmoid(X));
    
elseif partId == '2'
    disp(costFunction([0.25 0.5 -0.5]', X, y));
    
elseif partId == '3'
    [cost, grad] = costFunction([0.25 0.5 -0.5]', X, y);
    disp(cost);
    disp(grad);
    
elseif partId == '4'
    disp(predict([0.25 0.5 -0.5]', X));
    
elseif partId == '5'
    disp(costFunctionReg([0.25 0.5 -0.5]', X, y, 0.1));
    
elseif partId == '6'
    [cost, grad] = costFunctionReg([0.25 0.5 -0.5]', X, y, 0.1);
    disp(cost);
    disp(grad);
    
end
end