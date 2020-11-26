function output(partId)
% Random Test Cases
x1 = sin(1:10)';
x2 = cos(1:10)';
ec = 'the quick brown fox jumped over the lazy dog';
wi = 1 + abs(round(x1 * 1863));
wi = [wi ; wi];
if partId == '1'
    sim = gaussianKernel(x1, x2, 2);
    disp(sim);
elseif partId == '2'
    load('ex6data3.mat');
    [C, sigma] = dataset3Params(X, y, Xval, yval);
    disp(C);
    disp(sigma);
elseif partId == '3'
    word_indices = processEmail(ec);
    disp(word_indices);
elseif partId == '4'
    x = emailFeatures(wi);
    disp(x);
end
end
