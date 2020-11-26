function output(partId)
% Random Test Cases
n_u = 3; n_m = 4; n = 5;
X = reshape(sin(1:n_m*n), n_m, n);
Theta = reshape(cos(1:n_u*n), n_u, n);
Y = reshape(sin(1:2:2*n_m*n_u), n_m, n_u);
R = Y > 0.5;
pval = [abs(Y(:)) ; 0.001; 1];
Y = (Y .* double(R));  % set 'Y' values to 0 for movies not reviewed
yval = [R(:) ; 1; 0];
params = [X(:); Theta(:)];
if partId == '1'
    [mu, sigma2] = estimateGaussian(X);
    disp([mu(:); sigma2(:)]);
elseif partId == '2'
    [bestEpsilon, bestF1] = selectThreshold(yval, pval);
    disp([bestEpsilon(:); bestF1(:)]);
elseif partId == '3'
    [J] = cofiCostFunc(params, Y, R, n_u, n_m, n, 0);
    sprintf('%0.5f ', J(:));
elseif partId == '4'
    [J, grad] = cofiCostFunc(params, Y, R, n_u, n_m, n, 0);
    sprintf('%0.5f ', grad(:));
elseif partId == '5'
    [J] = cofiCostFunc(params, Y, R, n_u, n_m, n, 1.5);
    sprintf('%0.5f ', J(:));
elseif partId == '6'
    [J, grad] = cofiCostFunc(params, Y, R, n_u, n_m, n, 1.5);
    sprintf('%0.5f ', grad(:));
end
end
