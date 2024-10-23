function [c, a, f] = smv_1d_solver(x, zeta, varargin)


    [N, ~] = size(x);
    
    oversampling = varargin{1}; 
    num_iter = 10;
    tol = 1e-5;

    f_axis = linspace(0, 2*pi * (oversampling * N - 1) / oversampling / N, ...
        oversampling * N);

    x_fft = fft(x, N * oversampling , 1);
    [~, idx_max] = max(abs(x_fft));
    f = f_axis(idx_max);
    a_f = exp(1j*(0:1:N - 1)' * f);
    x_abs = conv(x, conj(flip(x)));
    x_abs = x_abs(N:end);
    x_abs(2:end) = x_abs(2:end);
    obj_old = sum(real(x_abs' * a_f), "all");

    for k = 1:num_iter 
        
        % Get Gradient and Hessian 
        f_grad = real(x_abs' * (a_f .* ((0:1:N-1)' * 1j)));
        f_hess = real(x_abs' * (a_f .* ((0:1:N - 1)' * 1j).^2));
        
        step_size = 1/f_hess;
        if step_size >= 0
            step_size = -1e-3;
        end

        % Perform the gradient descent 
        f_new = f - f_grad * step_size;

        % Calculate the New Objective Function
        a_f_new = exp(1j*(0:1:N - 1)' * f_new);
        obj_new = sum(real(x_abs' * a_f_new), "all");
        
        % Objectives Debugging 
        % disp([obj_old, obj_new, step_size])

        % Create some backtracking line-search 
        while(obj_new < obj_old)
            step_size = step_size/2;
            f_new = f - f_grad * step_size; 
            a_f_new = exp(1j*(0:1:N - 1)' * f_new);
            obj_new = sum(real(x_abs' * a_f_new), "all");

            if abs(step_size) <= 1e-5
                break;
            end
        end
        
        % Update Iterative Variables
        a_f = a_f_new;
        obj_old = obj_new;
        f = f_new;

        % Perform Thresholding 
        if f < 0
            f = f + 2 * pi;
        elseif f >= 2*pi
            f = f - 2 * pi;
        end

        % Termination Criterion
        if abs(step_size * f_grad) <= tol
            break;
        end

        % Some Debugging Information

    end

    a = a_f;

    % Get Iteration and thresholding 
    c = a_f' * x;
    a = a * c/abs(c);

    if abs(c) <= 1/zeta
        c = 0;
    else
        c = 1/norm(a_f)^2 * (abs(c) - 1/zeta);
    end

    return 

end