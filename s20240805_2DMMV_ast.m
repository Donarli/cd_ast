%% Create ast function 
addpath("Functions")
clc;
clear all;

%%

M = 16;
N = 16;
N_u = 16;
N_v = 16;
L = 5;

seq_u = linspace(0, N_u - 1, N_u);
seq_v = linspace(0, N_v - 1, N_v);
[seq_v, seq_u] = meshgrid(seq_v, seq_u);

Y = (randn(M,N,L) + 1j * randn(M,N,L)) / sqrt(2);


mean_ze2 =  exp(sum(  log(  1/4 + L./(1:1:(L - 1))/4  )  ))*sqrt(pi) * L / 2* sqrt(N_u * N_v);
std_ze = sqrt(N_u * N_v * L - mean_ze2^2);

zeta = mean_ze2 + 4.5 * std_ze;
zeta = 1/zeta;


num_source = 5;
p = 10^(0/20) * sqrt(L);
u_source = rand(num_source, 1) * 2 * pi;
v_source = rand(num_source, 1) * 2 * pi;

p_source =  (randn(num_source, L) + 1j * randn(num_source, L))/sqrt(2);
p_source = p_source ./ vecnorm(p_source, 2, 2);
p_source = p * p_source;

for i = 1:num_source
    
    p_source(i, :) = p * p_source(i, :);
    
    Y = Y + exp( 1j * (seq_u * u_source(i) + seq_v * v_source(i))) .* ...
        reshape(p_source(i, :), 1, 1, []);
    
    end

%% Optimization algorithm 

opts = [];
opts.iterations = 200;

% Use Definite epsilon
% opts.epsilon = 1e-2;

% Use relative epsilon
opts.epsilon = 1e-2 * zeta * norm(Y,'fro');

opts.oversampling = 8;
opts.rank_1_solver = @(x, zeta, varargin) mmv_2d_solver(x, zeta, opts.oversampling);
opts.zeta = zeta;

[sol_hist, log_hist] = ast_solver_cd(Y, opts);



figure
subplot(1,2,1)
plot(log_hist.gap_hist)
grid on
set(gca, 'YScale', 'log')
subplot(1,2,2)
plot(log_hist.obj_hist(log_hist.obj_hist > 0))
grid on 


%% Visualization


n_points = 300;
u_axis = linspace(0, 2*pi*(n_points - 1)/n_points, n_points);
v_axis = linspace(0, 2*pi*(n_points - 1)/n_points, n_points);


figure
subplot(1,2,1)
pcolor(u_axis, v_axis, vecnorm(fft2(Y, n_points, n_points), 2, 3))
title("Spectrum of Input Samples")
hold on
% Add Extra Visualization For Estimated Results
estimated_parameters = cell2mat(sol_hist.parameters');
plot(estimated_parameters(:, 2), estimated_parameters(:, 1), 'ro','LineWidth',2)
shading interp
colorbar
subplot(1,2,2)
pcolor(u_axis, v_axis, vecnorm(fft2(sol_hist.residual, n_points, n_points), 2, 3))
hold on
% Add Extra Visualization For Estimated Results
estimated_parameters = cell2mat(sol_hist.parameters');
plot(estimated_parameters(:, 2), estimated_parameters(:, 1), 'ro','LineWidth',2)
title("Spectrum of Residual")
shading interp
colorbar()
grid on

