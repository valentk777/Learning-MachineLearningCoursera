function output(partId)
% Random Test Cases
X = reshape(3 * sin(1:1:30), 3, 10);
Xm = reshape(sin(1:32), 16, 2) / 5;
ym = 1 + mod(1:16,4)';
t1 = sin(reshape(1:2:24, 4, 3));
t2 = cos(reshape(1:2:40, 4, 5));
t  = [t1(:) ; t2(:)];
if partId == '1'
    [J] = nnCostFunction(t, 2, 4, 4, Xm, ym, 0);
    disp(J);
elseif partId == '2'
    [J] = nnCostFunction(t, 2, 4, 4, Xm, ym, 1.5);
    disp(J);
elseif partId == '3'
    disp(sigmoidGradient(X));
elseif partId == '4'
    [J, grad] = nnCostFunction(t, 2, 4, 4, Xm, ym, 0);
    disp(J);
    disp(grad);
elseif partId == '5'
    [J, grad] = nnCostFunction(t, 2, 4, 4, Xm, ym, 1.5);
    disp(J);
    disp(grad);
end
end
