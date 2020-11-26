function exercise()

% Load from ex6data1:
% You will have X, y in your environment
load('ex6data1.mat');
% Plot training data
%plotData(X, y);

%C = 1;
%model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
%visualizeBoundaryLinear(X, y, model);

x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
fprintf('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 0.5 : \n\t%f\n', sim);

end