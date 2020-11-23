clc, close, clear;

save_tex = true;


features = ["steps","calories"];
resultspath = fullfile('..', 'results', join(features, '_'));
resultsdir = dir(resultspath);
timestamps = resultsdir([resultsdir.isdir]);
ts = zeros(size(timestamps,1)-2);
for i = 3:size(timestamps,1)
    ts(i) = str2double(timestamps(i).name);
end
resultspath = fullfile(resultspath, num2str(max(ts)));
addpath(resultspath);


accuracy_kNN = importdata("accuracy_kNN.csv");
accuracy_SVM = importdata("accuracy_SVM.csv");
accuracy_KDE = importdata("accuracy_KDE.csv");

N_range = 1:1:length(accuracy_kNN);

figure(1);
set(0,'defaultTextInterpreter','latex');
set(gcf,'Position', [480, 320, 480, 320]);
plot(N_range, accuracy_kNN, '.-', 'DisplayName', 'kNN, $k=1$');
hold on;
plot(N_range, accuracy_SVM, '.-', 'DisplayName', 'SVM, $C=10^4$, $\sigma=1$');
hold on;
plot(N_range, accuracy_KDE, '.-', 'DisplayName', 'KDE, $h=0.5$');
hold on;
plot(N_range, 1./N_range, '--', 'Color', [0.5, 0.5, 0.5], 'DisplayName', 'Naive');
hold off;
set(gca,'GridLineStyle',':');
grid;
legend('Location', 'southeast');
xlim([min(N_range), max(N_range)]);
xlabel('Number of users');
ylabel('P[success]');

if(save_tex)
    addpath('src/');
    outfile = char(fullfile('tex', join(features, '_'), 'plot_var_N.tex'));
    matlab2tikz(outfile);
end