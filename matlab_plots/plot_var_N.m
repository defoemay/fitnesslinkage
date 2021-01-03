clc, close, clear;

save_tex = true;
custom_ts = true;


features = ["steps","calories"];
resultspath = fullfile('..', 'results', join(features, '_'));

if custom_ts
    resultspath = fullfile(resultspath, "1607220075");
else    
    resultsdir = dir(resultspath);
    timestamps = resultsdir([resultsdir.isdir]);
    ts = zeros(size(timestamps,1)-2);
    for i = 3:size(timestamps,1)
        ts(i) = str2double(timestamps(i).name);
    end
    resultspath = fullfile(resultspath, num2str(max(ts)));
end

addpath(resultspath);

A = readtable("params.txt");

for i=1:size(A, 1)
    names(i) = strcat("", A(i,3).Var3{1}(1:end-1));
end

n_neighbors = str2double(A(names == 'n_neighbors', 4).Var4{1});
C = str2double(A(names == 'C', 4).Var4{1});
gamma = str2double(A(names == 'gamma', 4).Var4{1});
bandwidth = str2double(A(names == 'bandwidth', 4).Var4{1});

accuracy_kNN = importdata("accuracy_kNN.csv");
accuracy_SVM = importdata("accuracy_SVM.csv");
accuracy_KDE = importdata("accuracy_KDE.csv");

N_range = 1:1:length(accuracy_kNN);

figure(1);
set(0,'defaultTextInterpreter','latex');
set(gcf,'Position', [480, 320, 480, 320]);
str_kNN = strcat("", 'kNN, k=', num2str(n_neighbors));
str_SVM = strcat("", 'SVM, C=10^', num2str(floor(log10(C))),', \gamma=', num2str(gamma));
str_KDE = strcat("", 'KDE, h=', num2str(bandwidth));
plot(N_range, accuracy_kNN, '.-', 'DisplayName', str_kNN);
hold on;
plot(N_range, accuracy_SVM, '.-', 'DisplayName', str_SVM);
hold on;
plot(N_range, accuracy_KDE, '.-', 'DisplayName', str_KDE);
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

