clc
clear all;
y1 = [120.6; 132.2; 350.7; 385.1];
y2 = [2454.8,2394.5,2274.1,1299.9]; 
y3 = [56.99,39.68,57.27,77]; 
y4 = [131.3,138.9,147,155.7]; 
% 构造自变量矩阵
X = [y2; y3; y4]';
X = [ones(size(X, 1), 1) X];
coeffs = regress(y1, X);
% 提取系数
p = size(X, 2) / size(X, 1);
A = reshape(coeffs(1:end), [], p)';
disp(A)
