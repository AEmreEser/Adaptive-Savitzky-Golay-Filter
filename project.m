%% clear
clear all;

%% paper's algo's implementation

% A and H come from least squares regression
% this is formula 15 from the paper
% GUE-MSE (general unbiased estimate of mean squared error)
% !!!! n0 has to be the center of the window
function R = UNregularized_risk_estimate(M, n0, A, H, signal, sigma)
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

function R = regularized_risk_estimate(M, n0, A, H, signal, sigma)
    R_base = UNregularizedrisk_estimate(M, n0, A, H, signal, sigma);
    R = R_base + risk_regularizer(M, A, H, sigma);
end

function L = risk_regularizer(M, A, H, sigma)
    lambda = 12 * sigma.^2;
    L = (lambda / (2 * M + 1) ) * sum( diag( (A * H).^2 ) );
end

function R = estimate_risk(type, M, n0, A, H, signal, sigma)

    switch type
        case {'REGULAR','regularized', 'regular'}
            R = risk_regularizer(M, A, H, sigma) + UNregularized_risk_estimate(M, n0, A, H, signal, sigma);
        otherwise
            R = UNregularized_risk_estimate(M, n0, A, H, signal, sigma);
    end
end

function [R_min, p_opt] = select_opt_order(M, n0, signal, sigma, p_min, p_max, regularized)
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
        % TODO: SHOULD WE NOT BE PADDING THE SIGNAL BELOW???!!!
        switch regularized
            case {'regularized', 'REGULAR', 'regular'}
                Rs(p - p_min + 1) = estimate_risk('regular', M, n0+M, A, H, [zeros(1,M), signal, zeros(1,M)], sigma);
            otherwise
                Rs(p - p_min + 1) = estimate_risk('NONE', M, n0+M, A, H, [zeros(1,M), signal, zeros(1,M)], sigma);
        end % switch
        % disp(Rs);
    end

    [R_min, p_opt] = min(Rs); % p_opt is assigned the index of the min. R val's order, it is not necessarily the min order
    p_opt = p_opt + p_min - 1; % must add p_min to get the correct order, -1 since arrays start at index 1 in matlab
end

function [R_min, M_opt] = select_opt_filt_len(p, n0, signal, sigma, M_min, M_max, regularized)

    assert(2*M_min > p);
    assert(M_max > M_min);
    
    Rs = zeros(1, M_max - M_min + 1);
    
    for m = M_min : M_max
        % fprintf("inside opt filt, m: %d\n", m);
        [A, H] = construct_least_sq_matrices(m, p);
        % we pad the signal so that we can calculate the filter lengths at
        % the beginning and the end
        % we need to pass n0+m since we padded the signal
        switch regularized
            case {'regular', 'REGULAR', 'regularized'}
                Rs(m - M_min + 1) = estimate_risk('regular', m, n0+m, A, H, [zeros(1, m), signal, zeros(1, m)], sigma);
                % fprintf("Risk: %f\n", Rs(m - M_min + 1));
            otherwise
                Rs(m - M_min + 1) = estimate_risk('NONE', m, n0+m, A, H, [zeros(1, m), signal, zeros(1, m)], sigma);
        end % switch
    end

    [R_min, M_opt] = min(Rs); % M_opt is assigned the index where we ge tthe min R val. We need to add M_min to get the real
    % disp(Rs);
    % fprintf("M_opt: %d\n", M_opt);
    M_opt = M_opt + M_min - 1;
end

% this is the G-FL-O algo
function [R_min, p_opt, M_opt] = simult_optim(n0, signal, sigma, p_min, p_max, M_min, M_max, regularized)

    assert(2*M_min > p_max);
    assert(M_max > M_min);
    assert(p_max > p_min);
    
    Rs = zeros(p_max - p_min + 1, M_max - M_min +1);

    for p = p_min : p_max
        for m = M_min : M_max

            [A, H] = construct_least_sq_matrices(m, p);
            switch regularized
                case {'regularized', 'REGULAR', 'regular'}
                    Rs(p - p_min + 1, m - M_min + 1) = estimate_risk('regular', m, n0+m, A, H, [zeros(1,m), signal, zeros(1,m)], sigma);
                otherwise
                    Rs(p - p_min + 1, m - M_min + 1) = estimate_risk('NONE', m, n0+m, A, H, [zeros(1,m), signal, zeros(1,m)], sigma);
            end
            % fprintf("p: %d, m: %d, R: %d\n", p, m, Rs(p - p_min + 1, m - M_min + 1));
        end % for: M
    end % for: p

    [R_min, min_flat_idx] = min( Rs(:) ); % this get shte flattened vector's index
    [p_opt, M_opt] = ind2sub(size(Rs), min_flat_idx); % we convert it to matrix index
    p_opt = p_opt + p_min -1;
    M_opt = M_opt + M_min -1;
end

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
        in = xn(1, i : M+M+i);
        in = in(:);
        y(1,i) = H(1,:) * in; % convolution of the sgfilter's impulse response with the signal values in each window
        % we only need the first row of H -> check page 4 on the article
        % (right column)
        % H(1,:) is the flipped/time-reversed impulse response of the SG filter
    end

end

% applies at a single point only
% must pass the padded signal
function y = apply_sg(n0, extended_signal, H, M)

    window = extended_signal(n0-M : n0+M);
    window = window(:);
    y = H(1,:) * window;
end

% A and H calculation part of 
% from https://www.mathworks.com/matlabcentral/answers/335433-how-to-implement-savitzky-golay-filter-without-using-inbuilt-functions
function [A, H] = construct_least_sq_matrices(M, p)
    d = [-M : M]';
    l = length(d);
    A = zeros(l,p+1); % 2M + 1 many rows, p+1 many cols
    A(:,1) = 1;
    for i = 1:p
        A(:,i+1) = d(:,1).^i;
    end
    % disp(A)
    % disp(size(A))
    
    H = pinv(A' * A) * A';
    % disp(size(H))
end

% noise is quoted in terms of both snr and sigma (std dev) in the article
function sigma = sigma_from_snr(signal, snr_db)
    signal_power = mean(signal(:).^2);
    snr_linear = 10^(snr_db / 10);
    noise_power = signal_power / snr_linear;
    sigma = sqrt(noise_power);
end

function snr = get_snr(clean_signal, noisy_signal)
    if length(clean_signal) ~= length(noisy_signal)
        error('signals must be the same length');
    end

    noise = noisy_signal - clean_signal;
    signal_power = mean(clean_signal.^2);
    noise_power  = mean(noise.^2);
    snr = 10 * log10(signal_power / noise_power);
end

%% test generation
N = 200;
t = linspace(0, 2*pi, N);

true_signal = zeros(1,N);

fprintf("Poly orders:\n");
poly_orders = [0,7,1,6,2,5,3,4] % abrupt changes case
% poly_orders = [0,1,2,3,4,5,6,7] % gradual changes case
% poly_orders = [3]
segment_length = N / length(poly_orders);

for i = 1:length(poly_orders)
    start_idx = round((i-1) * segment_length) + 1;
    end_idx = round(i * segment_length);
    t_segment = t(start_idx:end_idx);
    true_signal(start_idx:end_idx) = polyval(polyfit(t_segment, sin(2*t_segment) + cos(3*t_segment), poly_orders(i)), t_segment);
end


% THESE FUNCTIONS DO NOT ADD THE GENERATED NOISE TO THE TRUE SIGNAL!!!
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
noise_type = 'gs';
% noise = sigma * randn(size(true_signal));
% [gauss_noise, sigma] = gen_gauss_noise(true_signal, snr);
[noise, sigma] = gen_noise(noise_type, true_signal, snr);
noisy_signal = true_signal + noise;


% OBSERVATIONS:
% regularized outperforms all in this case with uniform noise:
% [0,7,1,6,2,5,3,4] (other than that it almost always is behind
% unregularized. 
% unregularized performs best in laplacian noise'
% unregularized always is better than base sg
% under very specific conditions regularized might perform the worst.
% regularized is less sensitive to decreases in the snr (more noise per
% signal)


%% main

fprintf("Results for snr: %d, noise type: %s, num different orders in signal: %d\n", snr, noise_type, length(poly_orders));

%
% PARAMETERS:
% base sg params:
M = 12; p = 3; 
% optim params
p_min = 0; p_max = 7;
M_min = 4; M_max = 20;

%
% PLOTTING PARAMS
grid_r = 3; grid_c = 2; 


figure;
% orig signal
subplot(grid_r, grid_c, 1);
plot(t, true_signal);
title("original signal");

% nosiy signal
subplot(grid_r, grid_c, 3);
plot(t, noisy_signal);
title("noisy signal");

% sg filtered signal
subplot(grid_r, grid_c, 5);

[A, H, base_sg_res] = base_sg(noisy_signal, M, p);

plot(t, base_sg_res, 'r-');
hold on;
plot(t, true_signal, 'b-');
title(sprintf("base sg filtered signal (M: %d, p: %d)", M, p));
legend('sg fit', 'original signal');

% RISK ESTIMATES PLOT
Rs = zeros(1, length(noisy_signal));

% evaluate R at all centers of windows of size 2*M+1
for n0_i = [M+1:length(noisy_signal)-M]
    R = estimate_risk('NONE', M, n0_i, A, H, [zeros(1,M), noisy_signal, zeros(1, M)], sigma);
    Rs(n0_i) = R;
    % fprintf("Risk estimate for %d to %d: %d\n",n0-M, n0+M, R);
end

subplot(grid_r, grid_c, 2);
plot(t, Rs);
title(sprintf("UNregularized Risk estimates at different window centers (M: %d, p: %d)", M, p));

% OPTIMAL ORDER PLOT
opt_orders = zeros(1, length(noisy_signal));

for n0_i = [M+1:length(noisy_signal)-M]
    [~, order] = select_opt_order(M, n0_i, noisy_signal, sigma, p_min, p_max, 'NONE');
    opt_orders(n0_i) = order;
end

subplot(grid_r, grid_c, 4);
plot(t, opt_orders);
title("Optimal orders at different window centers (unregularized risk)");

% OPTIMAL LENGTHS PLOT

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

    [~, len] = select_opt_filt_len(p, n0_i, noisy_signal, sigma, M_min, M_max, 'NONE');
    opt_lens(n0_i) = len;
end

subplot(grid_r, grid_c, 6);
plot(t, opt_lens);
title("Optimal filter lengths at different window centers (unregularized risk)");

% % simultaneous optim
% sim_opt_lens = zeros(1, length(noisy_signal));
% sim_opt_ords = zeros(1, length(noisy_signal));
% 
% for n0_i = [1:length(noisy_signal)]
%     [~, p, m] = simult_optim(n0_i, noisy_signal, sigma, p_min, p_max, M_min, M_max, 'NONE');
%     sim_opt_ords(n0_i) = p;
%     sim_opt_lens(n0_i) = m;
%     % fprintf("p: %d, m: %d\n", p, m);
% end % for: n0
% 
% subplot(grid_r, grid_c, 7);
% plot(t, sim_opt_lens, "g");
% title("Simulatenously optimized filter lengths (unregularized risk)");
% 
% subplot(grid_r, grid_c, 8);
% plot(t, sim_opt_ords, "g");
% title("Simulatenously optimized orders (unregularized risk)");


% Base savitzky golay

figure
grid_r = 2; grid_c = 1;
plot(t, true_signal, "black", "LineWidth", 2);
hold on;
plot(t, base_sg_res, "red");
title("base savitzky-golay")
legend('base savitzky-golay', 'actual signal');

fprintf("MSE between base savitzky-golay and the TRUE signal:\n");
mse_base_sg = mse(true_signal, base_sg_res);
fprintf("\tMSE of base sg filtered signal: %f\n", mse_base_sg);

% The article has its own lingo for different configurations of the
% adaptive filter: 

% G-FL: fixed order, no regularization
%   M is elt [2, 20], p = 3
% G-FL-R: fixed order, with regularization
%   same as G-FL, lambda = 12 * sigma^2

% G-O: fixed length, no regularization
%   M = 15, p is elt [1, 5]
% G-O-R: fixed length, with regularization
%   same as G-O, lambda = 12 * sigma^2

% G-FL-O: simult. optim, no regularization
%   M is elt [10, 20], p is elt [1, 7]
% G-FL-O-R: simult. optim, with regularization
%   M is elt [10, 20], p is elt [1, 7], lambda = 12 * sigma^2

% G-FL-O(-R) - simultaneous optimization with unregularized risk

function [G_FL_O_filtered_signal, G_FL_O_orders, G_FL_O_lengths] = GFLO(noisy_signal, p_min, p_max, M_min, M_max, sigma)
    G_FL_O_filtered_signal = zeros(1, length(noisy_signal));
    G_FL_O_orders  = zeros(1, length(noisy_signal));
    G_FL_O_lengths = zeros(1, length(noisy_signal));
    
    for n0 = [1:length(noisy_signal)]
        [~, p, m] = simult_optim(n0, noisy_signal, sigma, p_min, p_max, M_min, M_max, 'NONE');
        [~, H] = construct_least_sq_matrices(m, p);
        G_FL_O_filtered_signal(n0) = apply_sg(n0+m, [zeros(1,m), noisy_signal, zeros(1,m)], H, m);
        G_FL_O_lengths(n0) = m;
        G_FL_O_orders(n0) = p;
    end
end % function

[G_FL_O_filtered_signal, G_FL_O_orders, G_FL_O_lengths] = GFLO(noisy_signal, p_min, p_max, M_min, M_max, sigma);

function err = mse(true_signal, signal)
    err = sum((true_signal - signal).^2) / (length(signal));
end

grid_r = 3; grid_c = 2;

figure;
subplot(grid_r, grid_c, 1);
plot(t, G_FL_O_filtered_signal, "r-");
hold on;
title("G-FL-O");
plot(t, true_signal, "black", "LineWidth", 2);
legend('G-FL-O', 'true signal');

subplot(grid_r, grid_c, 3);
plot(t, G_FL_O_orders);
title("G-FL-O orders");

subplot(grid_r, grid_c, 5);
plot(t, G_FL_O_lengths);
title("G-FL-O filter lengths");

fprintf("\nMSE between G-FL-O(-R) and the TRUE signal:\n");
mse_filtered = mse(true_signal, G_FL_O_filtered_signal);
fprintf("\tMSE of G-FL-O: %f\n", mse_filtered); % we see actual improvement here, nice : )

% G-FL-O-R - simultaneous optim with regularized risk
function [G_FL_O_R_filtered_signal, G_FL_O_R_orders, G_FL_O_R_lengths] = GFLOR(noisy_signal, p_min, p_max, M_min, M_max, sigma)
    G_FL_O_R_filtered_signal = zeros(1, length(noisy_signal));
    G_FL_O_R_orders  = zeros(1, length(noisy_signal));
    G_FL_O_R_lengths = zeros(1, length(noisy_signal));
    
    for n0 = [1:length(noisy_signal)]
        [~, p, m] = simult_optim(n0, noisy_signal, sigma, p_min, p_max, M_min, M_max, 'regular');
        [~, H] = construct_least_sq_matrices(m, p);
        G_FL_O_R_filtered_signal(n0) = apply_sg(n0+m, [zeros(1,m), noisy_signal, zeros(1,m)], H, m);
        G_FL_O_R_lengths(n0) = m;
        G_FL_O_R_orders(n0) = p;
    end
end % function

[G_FL_O_R_filtered_signal, G_FL_O_R_orders, G_FL_O_R_lengths] = GFLOR(noisy_signal, p_min, p_max, M_min, M_max, sigma);

subplot(grid_r, grid_c, 2);
plot(t, true_signal, "black", "LineWidth", 2);
hold on;
plot(t, G_FL_O_R_filtered_signal, "blue");
legend("true signal", "G-FL-O-R");
title("G-FL-O-R");

subplot(grid_r, grid_c, 4);
plot(t, G_FL_O_R_orders);
title("G-FL-O-R orders");

subplot(grid_r, grid_c, 6);
plot(t, G_FL_O_R_lengths);
title("G-FL-O-R filter lengths");

fprintf("\tMSE of G-FL-O-R: %f\n", mse(true_signal, G_FL_O_R_filtered_signal));

% TODO:
% visualize the boundaries between different orders of polynomials on the
% plots
% output snr vs input snr graphs for all algos
% report the results


% G-FL(-R) - Only optimize filter length
p = 3;

function [G_FL_filtered_signal, G_FL_filter_lens] = GFL(p, noisy_signal, M_min, M_max, sigma)
    G_FL_filtered_signal = zeros(1, length(noisy_signal));
    G_FL_filter_lens = zeros(1, length(noisy_signal));
    for n0 = [1:length(noisy_signal)]
        [~, m] = select_opt_filt_len(p, n0, noisy_signal, sigma, M_min, M_max, 'NONE');
        G_FL_filter_lens(n0) = m;
        [~, H] = construct_least_sq_matrices(m, p);
        G_FL_filtered_signal(n0) = apply_sg(n0+m, [zeros(1,m), noisy_signal, zeros(1,m)], H, m);
    end
end % function

[G_FL_filtered_signal, G_FL_filter_lens] = GFL(p, noisy_signal, M_min, M_max, sigma);

figure;
grid_r = 2;
grid_c = 2;

subplot(grid_r, grid_c, 1);
plot(t, true_signal, "black", "LineWidth", 2);
hold on;
plot(t, G_FL_filtered_signal, "red");
title("G-FL");
legend("true signal", "G-FL filtered");

subplot(grid_r, grid_c, 3);
plot(t, G_FL_filter_lens, "blue");
title("G-FL Filter lengths");


p = 3;
p_min = 1; p_max = 7;
M_min = 4; M_max = 20;

function [G_FL_R_filtered_signal, G_FL_R_filter_lens] = GFLR(p, noisy_signal, M_min, M_max, sigma)
    G_FL_R_filtered_signal = zeros(1, length(noisy_signal));
    G_FL_R_filter_lens = zeros(1, length(noisy_signal));

    for n0 = [1:length(noisy_signal)]
        [~, m] = select_opt_filt_len(p, n0, noisy_signal, sigma, M_min, M_max, 'regular');
        G_FL_R_filter_lens(n0) = m;
        [~, H] = construct_least_sq_matrices(m, p);
        G_FL_R_filtered_signal(n0) = apply_sg(n0+m, [zeros(1,m), noisy_signal, zeros(1,m)], H, m);
    end
end % function

[G_FL_R_filtered_signal, G_FL_R_filter_lens] = GFLR(p, noisy_signal, M_min, M_max, sigma);

subplot(grid_r, grid_c, 2);
plot(t, true_signal, "black", "LineWidth", 2);
hold on;
plot(t, G_FL_R_filtered_signal, "red");
title("G-FL-R");
legend("true signal", "G-FL-R filtered");

subplot(grid_r, grid_c, 4);
plot(t, G_FL_R_filter_lens, "blue");
title("G-FL-R Filter lengths");

fprintf("\nMSE between G-FL(-R) and the TRUE signal:\n");
fprintf("\tMSE of G-FL: %f\n", mse(true_signal, G_FL_filtered_signal));
fprintf("\tMSE of G-FL-R: %f\n", mse(true_signal, G_FL_R_filtered_signal));


% G-O(-R) - optimize polynomial order only

m = 15;
p_max = 5; p_min = 1;

function [G_O_filtered_signal, G_O_orders] = GO(m, noisy_signal, p_min, p_max, sigma)
    G_O_filtered_signal = zeros(1, length(noisy_signal));
    G_O_orders = zeros(1, length(noisy_signal));

    for n0 =  [1:length(noisy_signal)] % [m+1:length(noisy_signal)-m]
        [~, p] = select_opt_order(m, n0, noisy_signal, sigma, p_min, p_max, 'NONE');
        G_O_orders(n0) = p;
        [~, H] = construct_least_sq_matrices(m, p);
        G_O_filtered_signal(n0) = apply_sg(n0+m, [zeros(1,m), noisy_signal, zeros(1,m)], H, m);
    end
end

[G_O_filtered_signal, G_O_orders] = GO(m, noisy_signal, p_min, p_max, sigma);

figure;
subplot(grid_r, grid_c, 1);
plot(t, true_signal, "black", "LineWidth", 2);
hold on;
plot(t, G_O_filtered_signal, "red");
title("G-O");
legend("true signal", "G-O filtered signal");

subplot(grid_r, grid_c, 2);
plot(t, G_O_orders, "blue");
title("G-O orders");

function [G_O_R_filtered_signal, G_O_R_orders] = GOR(m, noisy_signal, p_min, p_max, sigma)
    G_O_R_filtered_signal = zeros(1, length(noisy_signal));
    G_O_R_orders = zeros(1, length(noisy_signal));
    
    for n0 = [1:length(noisy_signal)] % [m+1:length(noisy_signal)-m]
        [~, p] = select_opt_order(m, n0, noisy_signal, sigma, p_min, p_max, 'regular');
        G_O_R_orders(n0) = p;
        [~, H] = construct_least_sq_matrices(m, p);
        G_O_R_filtered_signal(n0) = apply_sg(n0+m, [zeros(1,m), noisy_signal, zeros(1,m)], H, m);
    end
end % function

[G_O_R_filtered_signal, G_O_R_orders] = GOR(m, noisy_signal, p_min, p_max, sigma);

subplot(grid_r, grid_c, 3);
plot(t, true_signal, "black", "LineWidth", 2);
hold on;
plot(t, G_O_R_filtered_signal, "red");
title("G-O-R");
legend("true signal", "G-O-R filtered signal");

subplot(grid_r, grid_c, 4);
plot(t, G_O_R_orders, "blue");
title("G-O-R orders");

fprintf("\nMSE between G-O(-R) and the TRUE signal:\n");
fprintf("\tMSE of G-O: %f\n", mse(true_signal, G_O_filtered_signal));
fprintf("\tMSE of G-O-R: %f\n", mse(true_signal, G_O_R_filtered_signal));

%%
% TESTING ON PIECEWISE SIGNALS WITH DIFFERENT SNRs
% SNR SWEEP PART

% noise_type = 'lp';
fprintf("\nDiffernt SNRs testing - noise type: %s\n", noise_type);

noise_types = ["un", "gs", "lp"];
noise_type_full_name = ["uniform", "gaussian", "laplace"];

for noise_type = noise_types
    
    input_snr_space = [-5:5:25]; 
    input_noise_space = zeros(length(input_snr_space), length(true_signal));
    noisy_signal_space = zeros(length(input_snr_space), length(true_signal));
    
    GO_output_snr_space = zeros(1, length(input_snr_space));
    GOR_output_snr_space = zeros(1, length(input_snr_space));
    
    GFL_output_snr_space = zeros(1, length(input_snr_space));
    GFLR_output_snr_space = zeros(1, length(input_snr_space));
    
    GFLO_output_snr_space = zeros(1, length(input_snr_space));
    GFLOR_output_snr_space = zeros(1, length(input_snr_space));
    
    for snr_i = 1:length(input_snr_space)
        [noise, sigma] = gen_noise(noise_type, true_signal, input_snr_space(snr_i));
        input_noise_space(snr_i, :) = noise;
        noisy_signal_space(snr_i, :) = input_noise_space(snr_i, :) + true_signal;
    
    end
    
    % disp(input_snr_space);
    % disp(noisy_signal_space);
    
    for snr_i = 1:length(input_snr_space)
        noisy_signal = noisy_signal_space(snr_i, :); 
        
        [GO_out, ~] = GO(m, noisy_signal, p_min, p_max, sigma);
        [GOR_out, ~] = GOR(m, noisy_signal, p_min, p_max, sigma);
        GO_output_snr_space(snr_i) = get_snr(true_signal, GO_out);
        GOR_output_snr_space(snr_i) = get_snr(true_signal, GOR_out);
    
        [GFL_out, ~] = GFL(p, noisy_signal, M_min, M_max, sigma);
        [GFLR_out, ~] = GFLR(p, noisy_signal, M_min, M_max, sigma);
        GFL_output_snr_space(snr_i) = get_snr(true_signal, GFL_out);
        GFLR_output_snr_space(snr_i) = get_snr(true_signal, GFLR_out);
    
        [GFLO_out, ~, ~] = GFLO(noisy_signal, p_min, p_max, M_min, M_max, sigma);
        [GFLOR_out, ~, ~] = GFLOR(noisy_signal, p_min, p_max, M_min, M_max, sigma);
        GFLO_output_snr_space(snr_i) = get_snr(true_signal, GFLO_out);
        GFLOR_output_snr_space(snr_i) = get_snr(true_signal, GFLOR_out);
    end
    
    figure;
    plot(input_snr_space, GO_output_snr_space, "r-o"); hold on; 
    plot(input_snr_space, GOR_output_snr_space, "g-s"); hold on; 
    plot(input_snr_space, GFL_output_snr_space, "b-x"); hold on; 
    plot(input_snr_space, GFLR_output_snr_space, "k-*"); hold on; 
    plot(input_snr_space, GFLO_output_snr_space, "m-^"); hold on; 
    plot(input_snr_space, GFLOR_output_snr_space, "c-d");
    
    legend("GO", "GOR", "GFL", "GFLR", "GFLO", "GFLOR");
    title(sprintf("output snr vs input snr for %s noise", noise_type_full_name( find(noise_types == noise_type)) ));
    xlabel("input snr (db)");
    ylabel("output snr (db)");

end % for noise type