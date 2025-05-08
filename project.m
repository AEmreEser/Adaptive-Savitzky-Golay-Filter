%% clear
clear all;
%% defs

% base savitzky golay implementation
% from https://www.mathworks.com/matlabcentral/answers/335433-how-to-implement-savitzky-golay-filter-without-using-inbuilt-functions
function [A, H, y] = base_sg(signal, M, p)
% refer IEEE paper of Robert Schafer 'What is Savitzky Golay Filter?' for better understanding.
    len = length(signal);
    xn = [zeros(1, M), signal, zeros(1, M)];
    y = zeros(1, len);
    
    [A, H] = construct_least_sq_matrices(M, p);
    
    for i = 1:len
        in = xn(1, i:M+M+i);
        in = in(:);
        y(1,i) = H(1,:) * in;% convolution of the sgfilter's impulse response with the signal values in each window
    end

end

% A and H calculation part of 
% from https://www.mathworks.com/matlabcentral/answers/335433-how-to-implement-savitzky-golay-filter-without-using-inbuilt-functions
function [A, H] = construct_least_sq_matrices(M, p)
    d = [-M : M]';
    l = length(d);
    A = zeros(l,p+1);
    A(:,1) = 1;
    for i = 1:p
        A(:,i+1) = d(:,1).^i;
    end
    % disp(A)
    % disp(size(A))
    
    H = pinv(A' * A) * A';
    % disp(size(H))
end

% noise func
function noise = generate_noise(type, sigma)
    
    sigma_sq = sigma^2;
    
    switch type
        case 'lp'
            noise = sigma * (randn(size(sigma_sq)) + randn(size(sigma_sq)));
            
        case 'un'
            a = -sqrt(3) * sigma;
            b = sqrt(3) * sigma;
            noise = a + (b - a) * rand(size(sigma_sq));
            
        otherwise
            assert(0); % fail
    end
end

% noise is quoted in terms of both snr and sigma (std dev) in the article
function sigma = sigma_from_snr(signal, snr_db)
    signal_power = mean(signal(:).^2);
    snr_linear = 10^(snr_db / 10);
    noise_power = signal_power / snr_linear;
    sigma = sqrt(noise_power);
end

%% test generation
N = 200;
t = linspace(0, 2*pi, N);

true_signal = zeros(1,N);

poly_orders = [2,0,1,5,1,7,3,4,0,1,0,2];
% poly_orders = [3];
segment_length = N / length(poly_orders);

for i = 1:length(poly_orders)
    start_idx = round((i-1) * segment_length) + 1;
    end_idx = round(i * segment_length);
    t_segment = t(start_idx:end_idx);
    true_signal(start_idx:end_idx) = polyval(polyfit(t_segment, sin(2*t_segment) + cos(3*t_segment), poly_orders(i)), t_segment);
end

function [noise, sigma] = gen_gauss_noise(true_signal, snr)
    signal_power = mean(true_signal.^2);
    noise_power = signal_power / (10^(snr/10));
    noise = sqrt(noise_power) * randn(size(true_signal));
    % sigma = std(noise);
    sigma = sqrt(noise_power); % more optimal
end

function [noise, sigma] = gen_lapl_noise(true_signal, snr)
    signal_power = mean(true_signal.^2);
    
    noise_power = signal_power / (10^(snr/10));
    
    sigma = sqrt(noise_power);
    
    % For Laplacian distribution, std = b*sqrt(2) 
    % where b is the scale parameter
    laplacian_scale = sigma / sqrt(2);
    
    u1 = rand(size(true_signal));
    u2 = rand(size(true_signal));
    noise = laplacian_scale * (log(u1) - log(u2));
end

function [noise, sigma] = gen_unif_noise(true_signal, snr)
    signal_power = mean(true_signal.^2);
    
    noise_power = signal_power / (10^(snr/10));
    
    sigma = sqrt(noise_power);
    
    % For uniform distribution in range [-a,a], std = a/sqrt(3)
    % So a = std*sqrt(3)
    uniform_range = sigma * sqrt(3);
    
    noise = (2*rand(size(true_signal)) - 1) * uniform_range;
end

function [noise, sigma] = gen_noise(type, true_signal, snr)

    switch type
        case 'gs'
            [noise, sigma] = gen_gauss_noise(true_signal, snr);
        case 'lp'
            [noise, sigma] = gen_lapl_noise(true_signal, snr);
        case 'un'
            [noise, sigma] = gen_unif_noise(true_signal, snr);
        otherwise
            assert(0); % fail
    end % switch

end


snr = 15; % db
% noise = sigma * randn(size(true_signal));
% [gauss_noise, sigma] = gen_gauss_noise(true_signal, snr);
[noise, sigma] = gen_noise('lp', true_signal, snr);
noisy_signal = true_signal + noise;

%% main

%
% PARAMETERS:
M_min = 10; M_max = 80;
M = 4; p = 3; n0 = M+1;
p_min = 0; p_max = 7;

grid_r = 6; grid_c = 1; 
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

[A, H, y] = base_sg(noisy_signal, M, p);

plot(t, y, 'r-');
hold on;
plot(t, true_signal, 'b-');
title("sg filtered signal ");
legend('sg fit', 'original signal');

% risk estimates plot
Rs = zeros(1, length(noisy_signal));

% evaluate R at all centers of windows of size 2*M+1
for n0_i = [M+1:length(noisy_signal)-M]
    R = risk_estimate(M, n0_i, A, H, [zeros(1,M), noisy_signal, zeros(1, M)], sigma);
    Rs(n0_i) = R;
    % fprintf("Risk estimate for %d to %d: %d\n",n0-M, n0+M, R);
end

subplot(grid_r, grid_c, 4);
plot(t, Rs);
title("Risk estimates at different window centers");

% optimal order plot
opt_orders = zeros(1, length(noisy_signal));

for n0_i = [M+1:length(noisy_signal)-M]
    [~, order] = select_opt_order(M, n0_i, noisy_signal, sigma, p_min, p_max);
    opt_orders(n0_i) = order;
end

subplot(grid_r, grid_c, 5);
plot(t, opt_orders);
title("Optimal orders at different window centers");

% optimal lengths plot

opt_lens = zeros(1, length(noisy_signal));

% we pass the actual order of the signal based on where n0 is
% this is an ideal assumption to see how the filter length selection
% performs
segment_progress = 1;
segment = 1;

% fprintf("length section\n");
for n0_i = [1:length(noisy_signal)]
    p = poly_orders(segment); % this part ensures that we fit with the correct p
    if segment_progress > segment_length
        segment = segment + 1;
        segment_progress = 1;
    end

    [~, len] = select_opt_filt_len(p, n0_i, noisy_signal, sigma, M_min, M_max);
    opt_lens(n0_i) = len;
end

subplot(grid_r, grid_c, 6);
plot(t, opt_lens);
title("Optimal filter lengths at different window centers");

clear all;

%% paper's algo's implementation

% A and H come from least squares regression
% this is formula 15 from the paper
function R = risk_estimate(M, n0, A, H, signal, sigma)
    % fprintf("dims A: %dx%d ", size(A,1), size(A,2));
    % fprintf("dims H: %dx%d ", size(H,1), size(H,2));
    % fprintf("dims signal: %dx%d\n", size(signal,1), size(signal,2));

    sigma_sq = sigma * sigma;
    wind_size = 2*M + 1;
    % fprintf("n0: %d, M: %d\n", n0, M);
    x = signal(n0-M:n0+M)'; % signal window transposed

    term1 = norm(A * H * x)^2;
    term2 = -2 * x' * H' * A' * x;
    term3 = 2 * sigma_sq * sum( diag(H' * A') );
    term4 = norm(x)^2;

    R = (1/wind_size) * (term1 + term2 + term3 + term4) - sigma_sq;
end

function [R_min, p_opt] = select_opt_order(M, n0, signal, sigma, p_min, p_max)
    % todo: use the arguments or whatever thing matlab has for param
    % checking
    assert(2*M > p_max); % require
    assert(p_max > p_min);

    Rs = zeros(1, p_max - p_min + 1);
    % disp(Rs)
    % p = p_min;

    for p = p_min : p_max
        % disp(p);
        [A, H] = construct_least_sq_matrices(M, p);
        Rs(p - p_min + 1) = risk_estimate(M, n0, A, H, signal, sigma);
        % disp(Rs);
    end

    [R_min, p_opt] = min(Rs); % p_opt is assigned the index of the min. R val's order, it is not necessarily the min order
    p_opt = p_opt + p_min - 1; % must add p_min to get the correct order, -1 since arrays start at index 1 in matlab
end

function [R_min, M_opt] = select_opt_filt_len(p, n0, signal, sigma, M_min, M_max)

    assert(2*M_min > p);
    assert(M_max > M_min);
    
    Rs = zeros(1, M_max - M_min + 1);
    
    for m = M_min : M_max
        [A, H] = construct_least_sq_matrices(m, p);
        % we pad the signal so that we can calculate the filter lengths at
        % the beginning and the end
        % we need to pass n0+m since we padded the signal
        Rs(m - M_min + 1) = risk_estimate(m, n0+m, A, H, [zeros(1, m), signal, zeros(1, m)], sigma);
    end

    [R_min, M_opt] = min(Rs); % M_opt is assigned the index where we ge tthe min R val. We need to add M_min to get the real
    % disp(Rs);
    % fprintf("M_opt: %d\n", M_opt);
    M_opt = M_opt + M_min - 1;
end