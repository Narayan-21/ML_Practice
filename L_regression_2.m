clear all, close all, clc

load hald;
A = ingredients;
b = heat;

[U, S, V] = svd(A,'econ');
x = V*inv(S)*U'*b;

plot(b,'k', 'LineWidth',2); hold on     % Plotting Data
plot(A*x, 'r-o', 'LineWidth', 1., 'MarkerSize',2);  % Plot Regression
l1 = legend('Heat Data', 'Regression')
set(l1, 'FontSize', 18)
grid on
set(gcf, 'Position', [1400 100 1500 1500])
set(gca, 'FontSize', 15)
xlabel('Ingredients')
ylabel('Temperature')
