clc, close, clear;

N = 20;
save_tex = true;
custom_ts = false;

resultspath = fullfile('..', 'results', 'split_pq');

if custom_ts
    resultspath = fullfile(resultspath, "1607220075");
else    
    resultsdir = dir(resultspath);
    timestamps = resultsdir([resultsdir.isdir]);
    ts = zeros(size(timestamps,1)-2);
    for i = 3:size(timestamps,1)
        ts(i) = str2double(timestamps(i).name);
    end
    max_ts = max(max(ts));
    resultspath = fullfile(resultspath, num2str(max_ts));
end

addpath(resultspath);

filepath = fullfile(resultspath, 'accuracy.csv');
acc = importdata(filepath);

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