function [c, a, params] = mmv_2d_solver(Y, zeta, varargin)


    [N_u, N_v, ~] = size(Y);

    seq_u = 0:1:N_u - 1;
    seq_v = 0:1:N_v - 1;
    [seq_v, seq_u] = meshgrid(seq_v, seq_u);
    
    oversampling = varargin{1}; 
    num_iter = 30;
    tol = 1e-8;

    % Create Some Basic Variables
    Yr = Y;

    % Create Some frequency axis
    f_u = linspace(0, 2*pi * (oversampling * N_u - 1) / oversampling / N_u, ...
        oversampling * N_u);
    f_v = linspace(0, 2*pi * (oversampling * N_v - 1) / oversampling / N_v, ...
        oversampling * N_v);

    % Create Initial frequency estimates 
    c_profile = squeeze(vecnorm(fft2(Yr, N_u * oversampling , N_v * oversampling),...
        2, 3));
    [~, idx_max] = max(abs(c_profile), [], 'all');
    [u_idx, v_idx] = ind2sub(size(c_profile), idx_max);
    f_u_max = f_u(u_idx);
    f_v_max = f_v(v_idx);

    a_f_u = exp(1j* seq_u * f_u_max);
    a_f_v = exp(1j* seq_v * f_v_max);
    a_f = a_f_u .* a_f_v;

    obj_old = norm(sum(Yr .* conj(a_f),[1,2]) ,"fro");

    for k = 1:num_iter 
        
        % Get Gradient and Hessian 
        
        a_f_du = (a_f_u * 1j) .* seq_u;
        a_f_d2u = - a_f_u .* (seq_u.^2); 
        a_f_dv = (a_f_v * 1j) .* seq_v;
        a_f_d2v = - a_f_v .* (seq_v.^2);

        % Calculate the Gradient 
        grad_f = zeros(2,1);

        Y_a_f =  sum(conj(Yr) .* a_f , [1,2]);
        Y_a_f_du =  sum(conj(Yr).*a_f_v.*a_f_du , [1,2]);
        Y_a_f_dv =  sum(conj(Yr).*a_f_dv.*a_f_u , [1,2]);
        Y_a_f_duv =  sum(conj(Yr).*a_f_dv.*a_f_du , [1,2]);
        Y_a_f_d2u =  sum(conj(Yr).*a_f_v.*a_f_d2u , [1,2]);
        Y_a_f_d2v =  sum(conj(Yr).*a_f_d2v.*a_f_u , [1,2]);

        grad_f(1) = 2 * sum(real(conj(Y_a_f) .* Y_a_f_du), "all");
        grad_f(2) = 2 * sum(real(conj(Y_a_f) .* Y_a_f_dv), "all");

        % Termination Criterion on Norm of Gradient
        if norm(grad_f, 'fro') <= tol
            break;
        end

        % Calculate the Hessian 
        H = zeros(2,2);
        H(1,1) = 2 * sum( real(conj(Y_a_f_du).* Y_a_f_du + conj(Y_a_f).*Y_a_f_d2u), "all");
        H(2,2) = 2 * sum( real(conj(Y_a_f_dv).* Y_a_f_dv + conj(Y_a_f).*Y_a_f_d2v), "all");
        H(2,1) = 2 * sum( real(conj(Y_a_f_dv).* Y_a_f_du + conj(Y_a_f).*Y_a_f_duv), "all");
        H(1,2) = H(2,1); 
        
        % Calculate the New Objective Function
        delta_uv = pinv(H) * grad_f;
        step_size = 1;
        f_u_max_new = f_u_max - step_size*delta_uv(1);
        f_v_max_new = f_v_max - step_size*delta_uv(2);
       
        a_f_u =  exp(1j*seq_u.*f_u_max_new);
        a_f_v =  exp(1j*seq_v.*f_v_max_new);
        a_f_new = a_f_u .* a_f_v;

        obj_new = squeeze(vecnorm(sum(Yr .* conj(a_f_new),[1,2]) ,2,3));

        % Objectives Debugging 
        % disp([norm(delta_uv) step_size])

        % Create some backtracking line-search 
        while(obj_new < obj_old)
            step_size = step_size/2;
            f_u_max_new = f_u_max - step_size*delta_uv(1);
            f_v_max_new = f_v_max - step_size*delta_uv(2);
           
            a_f_u =  exp(1j*seq_u.*f_u_max_new);
            a_f_v =  exp(1j*seq_v.*f_v_max_new);
            a_f_new = a_f_u .* a_f_v;

            obj_new = norm(sum(Yr .* conj(a_f_new),[1,2]) ,"fro");

            if abs(step_size) <= tol
                break;
            end
        end
        
        % Update Iterative Variables
        a_f = a_f_new;
        obj_old = obj_new;
        f_u_max = f_u_max_new;
        f_v_max = f_v_max_new;

        % Perform Thresholding 
        if f_u_max < 0
            f_u_max = f_u_max + 2 * pi;
        elseif f_u_max >= 2*pi
            f_u_max = f_u_max - 2 * pi;
        end

        if f_v_max < 0
            f_v_max = f_v_max + 2 * pi;
        elseif f_v_max >= 2*pi
            f_v_max = f_v_max - 2 * pi;
        end

    end

    a = a_f;

    % Get Iteration and thresholding 
    c = sum(Y .* conj(a), [1,2]);
    c_norm = norm(c, "fro");
    a = a .* c/c_norm;
    params = [f_u_max, f_v_max];

    if c_norm <= 1/zeta
        c = 0;
    else
        c = 1/norm(a_f, "fro")^2 * (c_norm - 1/zeta);
    end

    return 

end