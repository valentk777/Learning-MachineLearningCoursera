function example4()

% Initialization
clear;
% The following command loads the dataset. You should now have the variable X in your environment
load ('ex7data1.mat');
% Visualize the example dataset
figure;
plot(X(:, 1), X(:, 2), 'bo');
axis([0.5 6.5 2 8]); axis square;
hold on

% Before running PCA, it is important to first normalize X
[X_norm, mu, ~] = featureNormalize(X);
% Run PCA
[U, S] = pca(X_norm);
% Draw the eigenvectors centered at mean of data. These lines show the directions of maximum variations hold on;
drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
hold off;
fprintf('Top eigenvector U(:,1) = %f %f \n', U(1,1), U(2,1));

end