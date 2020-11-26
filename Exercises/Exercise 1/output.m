function output(partId)

% Random Test Cases
X1 = [ones(20,1) (exp(1) + exp(2) * (0.1:0.1:2))'];
Y1 = X1(:,2) + sin(X1(:,1)) + cos(X1(:,2));
X2 = [X1 X1(:,2).^0.5 X1(:,2).^0.25];
Y2 = Y1.^0.5 + Y1;

data = load('ex1data1.txt'); % read comma separated data
X_original = data(:, 1);
y = data(:, 2);

m = length(X_original); % number of training examples
X = [ones(m,1),data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
iterations = 1500;
alpha = 0.01;

if partId == '1'
    % plotData(X1, Y1)
    plotData(X, y)
    
elseif partId == '2'
    J = computeCost(X, y, theta);
    disp(J);
    
elseif partId == '3'
    J = computeCost(X, y, theta);
    disp(J);
    
    [theta, ~] = gradientDescent(X, y, theta, alpha, iterations);
    % plotData(1:size(J_history), J_history)
    disp(theta)
    
    % Compute and display initial cost with theta all zeros
    %J = computeCost(X, y, theta);
    %disp(J);
    
    % Compute and display initial cost with non-zero theta
    %J = computeCost(X, y,[-1; 2]);
    %disp(J);
    
    % Print theta to screen
    % Display gradient descent's result
    fprintf('Theta computed from gradient descent:\n%f,\n%f',theta(1),theta(2))
    
    % Plot the linear fit
    plotData(X_original, y)
    hold on; % keep previous plot visible
    plot(X_original, X*theta, '-')
    legend('Training data', 'Linear regression')
    hold off % don't overlay any more plots on this figure
    % Predict values for population sizes of 35,000 and 70,000
    predict1 = [1, 3.5] * theta;
    fprintf('For population = 35,000, we predict a profit of %f\n', predict1*10000);
    predict2 = [1, 7] * theta;
    fprintf('For population = 70,000, we predict a profit of %f\n', predict2*10000);
    
elseif partId == '4'
    [X_norm, ~, ~] = featureNormalize(X2(:,2:4));
    disp(X_norm)
    
elseif partId == '5'
    disp(computeCostMulti(X2, Y2, [0.1 0.2 0.3 0.4]'));
elseif partId == '6'
    % disp(gradientDescentMulti(X2, Y2, [-0.1 -0.2 -0.3 -0.4]', 0.01, 10));
    
    % Load Data
    data = load('ex1data2.txt');
    X = data(:, 1:2);
    y = data(:, 3);
    m = length(y);
    % Print out some data points
    % First 10 examples from the dataset
    fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
    
    % Scale features and set them to zero mean
    [X, ~, ~] = featureNormalize(X);
    
    % Add intercept term to X
    X = [ones(m, 1) X];
    % Run gradient descent
    % Choose some alpha value
    alpha = 0.1;
    num_iters = 400;
    % Init Theta and Run Gradient Descent
    theta = zeros(3, 1);
    [theta, ~] = gradientDescentMulti(X, y, theta, alpha, num_iters);
    % Display gradient descent's result
    fprintf('Theta computed from gradient descent:\n%f\n%f\n%f',theta(1),theta(2),theta(3))
    
    % Run gradient descent:
    % Choose some alpha value
    alpha = 1;
    num_iters = 50;
    % Init Theta and Run Gradient Descent
    theta = zeros(3, 1);
    [~, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
    % Plot the convergence graph
    plot(1:num_iters, J_history, '-b', 'LineWidth', 2);
    xlabel('Number of iterations');
    ylabel('Cost J');
    
elseif partId == '7'
    disp(normalEqn(X2, Y2));
    
    % Solve with normal equations:
    % Load Data
    data = csvread('ex1data2.txt');
    X = data(:, 1:2);
    y = data(:, 3);
    m = length(y);
    % Add intercept term to X
    X = [ones(m, 1) X];
    % Calculate the parameters from the normal equation
    theta = normalEqn(X, y);
    % Display normal equation's result
    fprintf('Theta computed from the normal equations:\n%f\n%f\n%f', theta(1),theta(2),theta(3));
    
elseif partId == '8'
    % Visualizing J(theta_0, theta_1):
    % Grid over which we will calculate J
    theta0_vals = linspace(-10, 10, 100);
    theta1_vals = linspace(-1, 4, 100);
    % initialize J_vals to a matrix of 0's
    J_vals = zeros(length(theta0_vals), length(theta1_vals));
    % Fill out J_vals
    for i = 1:length(theta0_vals)
        for j = 1:length(theta1_vals)
            t = [theta0_vals(i); theta1_vals(j)];
            J_vals(i,j) = computeCost(X, y, t);
        end
    end
    
    % Because of the way meshgrids work in the surf command, we need to
    % transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = J_vals';
    % Surface plot
    figure;
    surf(theta0_vals, theta1_vals, J_vals)
    xlabel('\theta_0'); ylabel('\theta_1');
    % Contour plot
    figure;
    % Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
    xlabel('\theta_0'); ylabel('\theta_1');
    hold on;
    plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
    hold off;
end
end
