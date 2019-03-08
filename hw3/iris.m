x = csvread('iris.csv')
x_train = x(:, 1:4)
y_train = zeros(size(x(:, 1:3)));

for i = 1:length(x)
    y_train(i, :) = [0 0 0];
    y_train(i, int16(x(i, 5))+1) = 1;
end