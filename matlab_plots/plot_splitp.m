clc, close, clear;

N = 20;
save_tex = true;

resultspath = fullfile('..', 'results', 'split_p');

addpath(resultspath);

acc = importdata(strcat('accuracy_N', num2str(N), '.csv'));

p_range = acc(:, 1);

figure(1);
set(0,'defaultTextInterpreter','latex');
set(gcf,'Position', [480, 320, 480, 320]);
plot(p_range, acc(:, 2), '.-', 'DisplayName', 'kNN');
hold on;
plot(p_range, acc(:, 3), '.-', 'DisplayName', 'SVM');
hold on;
plot(p_range, acc(:, 4), '.-', 'DisplayName', 'KDE');
hold on;
plot(p_range, 1/N*ones(size(p_range)), '--', 'Color', [0.5, 0.5, 0.5], 'DisplayName', 'Naive');
hold off;
set(gca,'GridLineStyle',':');
grid;
legend('Location', 'southeast');
xlim([p_range(1), p_range(end)]);
xlabel('Fraction of data used for training');
ylabel('P[success]');

if(save_tex)
    addpath('src/');
    outfile = char(fullfile('tex', strcat('splitp_N', num2str(N) ,'.tex')));
    matlab2tikz(outfile);
end