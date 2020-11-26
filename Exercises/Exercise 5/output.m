function output(partId)
% Random Test Cases
X = [ones(10,1) sin(1:1.5:15)' cos(1:1.5:15)'];
y = sin(1:3:30)';
Xval = [ones(10,1) sin(0:1.5:14)' cos(0:1.5:14)'];
yval = sin(1:10)';
if partId == '1'
    [J] = linearRegCostFunction(X, y, [0.1 0.2 0.3]', 0.5);
    disp(J);
elseif partId == '2'
    [J, grad] = linearRegCostFunction(X, y, [0.1 0.2 0.3]', 0.5);
    disp(grad);
elseif partId == '3'
    [error_train, error_val] = learningCurve(X, y, Xval, yval, 1);
    disp([error_train(:); error_val(:)]);
elseif partId == '4'
    [X_poly] = polyFeatures(X(2,:)', 8);
    disp(X_poly);
elseif partId == '5'
    [lambda_vec, error_train, error_val] = validationCurve(X, y, Xval, yval);
    disp([lambda_vec(:); error_train(:); error_val(:)]);
end
end
