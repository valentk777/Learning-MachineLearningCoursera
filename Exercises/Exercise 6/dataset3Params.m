function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

C_tests = [0.01 0.03 0.1 0.3 1, 3, 10 30];
sigma_tests = [0.01 0.03 0.1 0.3 1, 3, 10 30];

[m,n] = ndgrid(C_tests,sigma_tests);
Comb = [m(:),n(:)];

prediction_error_min = Inf;

for col=1:size(Comb,1)
    C_test = Comb(col,1);
    sigma_test = Comb(col,2);
    
    model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
    predictions = svmPredict(model, Xval);
    prediction_error = mean(double(predictions ~= yval));
    
    if prediction_error < prediction_error_min
        prediction_error_min = prediction_error;
        C = C_test;
        sigma = sigma_test;
    end
end

end
