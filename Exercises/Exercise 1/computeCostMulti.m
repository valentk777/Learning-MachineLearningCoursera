function J = computeCostMulti(X, y, theta)
% J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
% parameter for linear regression to fit the data points in X and y

m = length(y); % number of training examples

h = X * theta;
J = ((h - y)' * (h - y)) / (2 * m); % Vectorized

end