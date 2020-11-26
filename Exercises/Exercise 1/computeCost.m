function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

m = length(y); % number of training examples

% Un-Vectorized
i = 1:m;
J = (1 / (2 * m)) * sum(((theta(1) + theta(2) .* X(i,2)) - y(i)) .^ 2);

end