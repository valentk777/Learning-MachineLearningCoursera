function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

m = length(y); % number of training examples
h_theta = sigmoid(X * theta);
regular = lambda / (2 * m) * (theta(2:length(theta)))' * theta(2:length(theta));

% J, grad = costFunction(theta, X, y)
J = (y' * log(h_theta) + (1-y)' * log(1-h_theta)) / -m;
J = J + regular;

% version 1
% first without theta
thetaZero = theta;
thetaZero(1) = 0;

grad = ((1 / m) * (h_theta - y)' * X) + (lambda / m) * thetaZero';

% version 2
% grad(1,1) = ((h-y)'*X(:,1))./m;
% grad(2:end,1)=(((h-y)'*X(:,2:end))./m)' + (lambda./m)*theta(2:end);

end