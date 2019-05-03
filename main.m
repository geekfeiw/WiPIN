clc;
clear;
close all;

%% load csi and only use amplitude, num_sample \times 30, where 30 is subcarrier number
load('sample.mat', 'ori_csi');
ori_csi = abs(ori_csi);

%% plot original csi
figure 
subplot(2,1,1), plot(ori_csi);
title('original csi')

%% Butterworth filter, 5th-order;
% Normalized-low-pass:(2/100) * sampling rate:500Hz -> 10Hz
[b, a] = butter(5, 2/100);
flt_csi = filter(b,a,ori_csi);

subplot(2,1,2), plot(flt_csi);
title('filted csi')

%% flt_csi largely distored in first 2000 samples, neglect them
flt_csi = flt_csi(2000:end,:);
% average every 100 samples as one training/test instance
flt_csi = meanfilter(flt_csi, 100);


%% multi-path efect mitigation, take one sample as example
% filted csi --ifft-> time domain (TD) --fft-> frequency domain (FD)
flt_csi_first = flt_csi(1,:);
flt_csi_TD = ifft(flt_csi_first);
figure;
subplot(2,2,1), 
bar(1, abs(flt_csi_TD(1))), hold on; 
bar([2:30], abs(flt_csi_TD(2:30)));
title('csi time domain')

% keep the shortest paths, suppress the remaining
flt_csi_TD(2:30) = flt_csi_TD(2:30)/1000;
subplot(2,2,3), 
bar(1, abs(flt_csi_TD(1))), hold on; 
bar([2:30], abs(flt_csi_TD(2:30)));
title('suppress csi from shortest paths')

flt_csi_FD = fft(flt_csi_TD);

subplot(2,2,2),
scatter([1:30], flt_csi_first), hold on;
plot([1:30], flt_csi_first, '--')
title('original csi in amplitude')

flt_csi_first_ifft_fft = abs(flt_csi_FD);

subplot(2,2,4),
scatter([1:30], flt_csi_first_ifft_fft), hold on;
plot([1:30], flt_csi_first_ifft_fft, '--')
title('csi amplitude after ifft&fft'); hold off;

%% do above ifft/fft over all samples
for i = 1:length(flt_csi)
    temp = ifft(flt_csi(i,:));
    temp(2:30) = temp(2:30)/1000;
    temp = fft(temp);
    
    flt_csi(i,:) = abs(temp);
end



%%
features = zeros(length(flt_csi),39);
for i = 1:length(flt_csi)
    features(i,1:30) = flt_csi(i,:);
    
    features(i,31) = mean(flt_csi(i,:));
    features(i,32) = std(flt_csi(i,:));
    features(i,33) = mad(flt_csi(i,:));
    features(i,34) = mad(flt_csi(i,:),1);
    features(i,35) = max(flt_csi(i,:));
    features(i,36) = min(flt_csi(i,:));
    features(i,37) = skewness(flt_csi(i,:));
    features(i,38) = kurtosis(flt_csi(i,:));
    features(i,39) = xentropy(flt_csi(i,:), 10); % 10 bins entropy
    
end


%% !!!Please note that, following codes just for demostrating the usage of liblibnear
%%% data preparation randomly
% random labeling [0,...,N-1]; N-class classification
N = 3;
label = floor(N*rand(length(features),1));
index = randperm(length(features));

% random 80 percents as training
train_data = features(index(1:floor(0.8*length(features))) ,:);
train_label = label(index(1:floor(0.8*length(features))),1);
% random 20 percents as test 
test_data = features(index(ceil(0.8*length(features)):end) ,:);
test_label = label(index(ceil(0.8*length(features)):end) ,1);

% normailze every dimension of train data to [-1,+1]
train_data = train_data';
[train_data_norm, pattern] = mapminmax(train_data,-1,1);
train_data_norm = train_data_norm';
% apply the maximum/minimum from training data to normalize test data
test_data = test_data';
test_data_norm = mapminmax('apply',test_data,pattern);
test_data_norm = test_data_norm';

%%% training classifiers with Liblinear
%%% the way to install liblinear
% cd to liblinear_folder/matlab
% mex -setup
% make
%%%
liblinearPath = 'liblinear-2.30/matlab';
addpath(liblinearPath)

% only classifiers

% typing train in matlab terminal to list parameters to set svm 
% -s type: 2 -- L2-regularized L2-loss support vector classification (primal)
% -B bias : if bias >= 0, instance x becomes [x; bias]
model = train(train_label, sparse(train_data_norm), '-s 2 -B 1');

% y = wx + bias
% model.w -> num_class \times [dim_features; bias]
% the index of maximum is the prediction
test_predict = test_data_norm * model.w(:,1:end-1)' +  repmat(model.w(:,end)', [length(test_label), 1]);
[~, idx] = max(test_predict,[], 2);
accuracyx = sum((idx-1)==test_label)/length(test_label);

% test_predict: num_test \times num_class
confidence_softmax = exp(test_predict)./repmat(sum(exp(test_predict),2), [1, length(unique(train_label))]); 

figure;
correct_index = find((idx-1)==test_label);
plot(max(confidence_softmax(correct_index,:),[], 2));
title('correct confidence in test');

% If not compute prediction confidences
% accuracy can also be computed by the 'predict' function of liblinear 
[predicted_label,accuracy,~ ] = predict(test_label, sparse(test_data_norm), model);
% accuracy = sum(predicted_label==test_label)/length(test_label);


rmpath(liblinearPath)


%% mean average pooling with the stride
function out = meanfilter(in, stride)
    out = zeros(floor(length(in)/stride), 30);
    
    for i = 1: floor(length(in)/stride)
        out(i,:) = mean(in(stride*i-stride+1: stride*i, :) , 1);  
    end

end


%% 10 bin entropy
function out = xentropy(in, bin_number)
    mini_val = min(in);
    max_val = max(in);
    % bins from mini to max
    bins = linspace(mini_val, max_val, bin_number);
    
    out = 0;
    for i = 1:length(bins)-1    
        upx = in>=bins(i); % how many larger than bins(i)
        lowx = in<bins(i+1); % how many smaller than bins(i+1)
        count = sum(upx&lowx); % count those meet above two
        if count>0
            p = count/length(in);
            out = -p*log(p) + out;
        end
    end

end



