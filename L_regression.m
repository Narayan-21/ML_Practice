% Linear Regression using SVD - Example #1

clear all, close all, clc

x=3; % True Slope
a = [-2:0.1:2]';
b = a*x + 1*randn(size(a));

plot(a, x*a,'k', 'LineWidth',2);
hold
plot(a,b,'rx','LineWidth',2);

[U,S,V] = svd(a, 'econ');
xtilde = V*inv(S)*U'*b;

plot(a, xtilde*a, 'b--', 'LineWidth',2)
l1 = legend('True line', ' Noisy Data', 'Regression line');
set(l1, 'Location', 'NorthWest')
set(l1, 'FontSize', 18)
grid on
set(gcf, 'Position', [1400 100 1500 1500])
set(gcf, 'PaperPositionMode', 'auto')
set(gca, 'FontSize', 15)
xlabel('a')
ylabel('b')
