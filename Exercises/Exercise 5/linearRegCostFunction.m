function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

h_theta = X * theta;

regular = lambda / (2 * m) * (theta(2:length(theta)))' * theta(2:length(theta));

% J, grad = costFunction(theta, X, y)
J = (h_theta - y)' * (h_theta - y) / (2*m);
J = J + regular;

% version 1
% first without theta
thetaZero = theta;
thetaZero(1) = 0;

grad = ((1 / m) * (h_theta - y)' * X) + (lambda / m) * thetaZero';

grad = grad(:);

end
