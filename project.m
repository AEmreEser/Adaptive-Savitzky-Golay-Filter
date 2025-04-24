%% clear
clear all;
%% defs

% base savitzky golay implementation
% from https://www.mathworks.com/matlabcentral/answers/335433-how-to-implement-savitzky-golay-filter-without-using-inbuilt-functions
function [A, H, y] = base_sg(signal,M,p)
% refer IEEE paper of Robert Schafer 'What is Savitzky Golay Filter?' for better understanding.
    len=length(signal);
    xn=[zeros(1,M),signal,zeros(1,M)];
    y=zeros(1,len);
    d=[-M:M]';
    l=length(d);
    A=zeros(l,p+1);
    A(:,1)=1;
    for i=1:p,
        A(:,i+1)=d(:,1).^i;
    end
    disp(A)
    disp(size(A))
    
    H=pinv(A'*A)*A';% fliplr(H(1,:)) is actually the impulse response of the savitzky-golay filter.
    disp(size(H))
    
    for i=1:len,
        in=xn(1,i:M+M+i);
        in=in(:);
        y(1,i)=H(1,:)*in;% convolution of the sgfilter's impulse response with the signal values in each window
    end

end

%% test generation
N = 200;
t = linspace(0, 2*pi, N);

true_signal = zeros(1,N);

poly_orders = [2,7,3,4,0,1,0,2];
% poly_orders = [3];
segment_length = N / length(poly_orders);
for i = 1:length(poly_orders)
    start_idx = round((i-1) * segment_length) + 1;
    end_idx = round(i * segment_length);
    t_segment = t(start_idx:end_idx);
    true_signal(start_idx:end_idx) = polyval(polyfit(t_segment, sin(2*t_segment) + cos(3*t_segment), poly_orders(i)), t_segment);
end

p_min = 0; p_max = 7;
M_min = 10; M_max = 80;

% noise
% gaussian -- NOISE MUST NOT BE GAUSSIAN, THAT IS THE WHOLE POINT OF THE
% ARTICLE!!!!!!
sigma = 0.3;
noise = sigma * randn(size(true_signal));
noisy_signal = true_signal + noise;

% todo: laplacian

%% main
grid_r = 5; grid_c = 1; 
figure;
% orig signal
subplot(grid_r, grid_c, 1);
plot(t, true_signal);
title("original signal");

% nosiy signal
subplot(grid_r, grid_c, 2);
plot(t, noisy_signal);
title("noisy signal");

% sg filtered signal
subplot(grid_r, grid_c, 3);

M = 4; p = 3; n0 = M+1;
[A, H, y] = base_sg(noisy_signal, M, p);

plot(t, y);
hold on;
plot(t, true_signal, 'r-');
title("sg filtered signal ");
legend('sg fit', 'original signal');

Rs = zeros(length(noisy_signal));

% evaluate R at all centers of windows of size 2*M+1
for n0_i = [M+1:length(noisy_signal)-M]
    R = risk_estimate(M, n0_i, A, H, noisy_signal, sigma);
    Rs(n0_i) = R;
    % fprintf("Risk estimate for %d to %d: %d\n",n0-M, n0+M, R);
end

subplot(grid_r, grid_c, 4);
plot(t, Rs);
title("Risk estimates at different window centers");

clear all;

%% paper's algo's implementation

% A and H come from least squares regression
% todo: use x instead of signal and n0??
function R = risk_estimate(M, n0, A, H, signal, sigma)
    % fprintf("dims A: %dx%d ", size(A,1), size(A,2));
    % fprintf("dims H: %dx%d ", size(H,1), size(H,2));
    % fprintf("dims signal: %dx%d\n", size(signal,1), size(signal,2));

    sigma_sq = sigma * sigma;
    wind_size = 2*M + 1;
    x = signal(n0-M:n0+M)'; % signal window transposed

    term1 = norm(A * H * x)^2;
    term2 = -2 * x' * H' * A' * x;
    term3 = 2 * sigma_sq * sum( diag(H' * A') );
    term4 = norm(x)^2;

    R = (1/wind_size) * (term1 + term2 + term3 + term4) - sigma_sq;
end
