% 导入数据
y1 = [120.6, 132.2, 350.7, 385.1]; 
y2 = [2454.8,2394.5,2274.1,1299.9]; 
data = [y1', y2'];
% 计算协方差矩阵
sigma = cov(data);
% 提取y1和y2的平均值
mu_y1 = mean(y1);
mu_y2 = mean(y2);
% 设置滞后阶数
p = 1;
% 初始化变量
F = zeros(p, p);
f = zeros(p, 1);
% 计算因果关系
for k = 1:p
    for j = 1:p
        if k > j
            continue;
        else
            F(k, j) = sigma(1, 1) / sigma(1, 1+j-k);
            f(k) = f(k) + F(k, j);
        end
    end
end
% 显示结果
disp(['f = ', num2str(f)]);
if f > 1
    disp('y2 granger causes y1');
else
    disp('y1 granger causes y2');
end
