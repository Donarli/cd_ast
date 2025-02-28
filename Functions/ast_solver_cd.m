function [sol, log] = ast_solver_cd(y, opts)


    % Set options if missing 
    opts = set_default_opts_ast(opts, y);

    % Define solution of the solver
    sol = [];
    sol.atoms = {};
    sol.scalars = [];
    sol.parameters = {};
    

    % Define logging of the solver
    log = [];
    log.obj_hist = zeros(opts.iterations, 1);
    log.obj_real = zeros(opts.iterations, 1);
    log.gap_hist = zeros(opts.iterations, 1);

    % See if initial solution is provided 
    a_coef = opts.init_atoms;
    f_coef = opts.init_params;

    c_coef = opts.init_scalars; 
    yr = opts.init_yr;
    zeta = opts.zeta;


    % Get a inner product 
    inner_xy = opts.inner_product;

    % Some Sanity Check
    if numel(a_coef) ~= length(c_coef)
        error("Incorrect Matching!")
    end
    
    x = 0;
    for i = 1:numel(a_coef)
        x = x + a_coef{i} * c_coef(i);
    end
    % Compute the Duality Gap 
    dual_gap = abs(sum(abs(c_coef), "all") - zeta * inner_xy(yr, x));

    norm_y = inner_xy(y, y);
    if  inner_xy(x + yr, y) ~= norm_y 
        error("Incorrect Residual!")
    end

    % Compute the objective 
    obj_k = zeta / 2 * inner_xy(yr, yr) + sum(abs(c_coef), "all");

    L = numel(c_coef); 
    i = 1;

    % Get a delta factor ready 
    delta = opts.epsilon / (opts.zeta * norm_y + opts.epsilon);

    % Zeta Prime
    zeta_prime = zeta / (1 - delta);

    for k = 1:opts.iterations

        if i <= L             
            
            % Get the refinement vector 
            yr_i = yr + c_coef(i) * a_coef{i};
            
            % Get the projection ready 
            [c_i, a_i, params_i] = opts.rank_1_solver(yr_i, zeta_prime, f_coef{i});

            % Get the y-vector back 
            yr = yr_i - c_i * a_i;
            
            % Adjust the indices 
            if abs(c_i) == 0
                c_coef(i) = [];
                a_coef(i) = [];
                f_coef(i) = [];

                L = L - 1;
                i = i - 1;
            else
                c_coef(i) = c_i;
                a_coef{i} = a_i;
                f_coef{i} = params_i;
            end
            
            % Increase index by one 
            i = i + 1;

        else
            % Get the x ready again for computing the duality gap 
            x = 0;
            for i = 1:numel(a_coef)
                x = x + a_coef{i} * c_coef(i);
            end

            % Compute the Objective 
            obj_k = zeta/2 * inner_xy(y - x, y - x) + sum(abs(c_coef), "all");
            obj_k_real = zeta_prime/2 * inner_xy(y - x, y - x) + sum(abs(c_coef), "all");

            % Compute the Duality Gap 
            dual_gap = abs(sum(abs(c_coef), "all") - zeta * inner_xy(yr, x));

            if dual_gap <= opts.epsilon 

                [c_i, ~, params_new] = opts.rank_1_solver(yr, zeta);

                % Detects whether a new element should be added
                if abs(c_i) > 0

                    [c_i, a_i, params_i] = opts.rank_1_solver(yr, zeta_prime, params_new);
                    a_coef{numel(a_coef) + 1} = a_i;
                    c_coef(numel(a_coef)) = c_i;
                    f_coef{numel(a_coef)} = params_i;


                    yr = yr - c_i * a_i; 
                    L = numel(a_coef);
                    i = 1;
                else
                    fprintf("Iteration Terminates wtih %d iterations !! \n", k);
                    break;
                end
            else
                i = 1;
            end
        end
        log.gap_hist(k) = dual_gap;
        log.obj_hist(k) = obj_k;
        log.obj_real(k) = obj_k_real;
        
        % % Debugging =======================================================
        % if k > 1 && (log.obj_real(k) > log.obj_real(k - 1))
        %     disp(['Wrong Objectives: ', num2str(log.obj_real(k)), '-> ',...
        %         num2str(log.obj_real(k - 1)) ])
        % end

    end

    sol.atoms = a_coef;
    sol.scalars = c_coef;
    sol.residual = yr;
    sol.parameters = f_coef;

    return 

end