function opts = set_default_opts_ast(opts, y)

    if ~isfield(opts, 'iterations')
        opts.iterations = 100;
    end

    if ~isfield(opts, 'epsilon')
        opts.epsilon = 1e-3;
    end

    if ~isfield(opts, 'rank_1_solver')
        error('Missing Legitmate Rank-1 Solver')
    end

    if ~isfield(opts, 'zeta') || (opts.zeta <= 0)
        error("Missing Legitmate Threshold!")
    end

    if ~isfield(opts, 'inner_prodct')
        opts.inner_product = @(x,y) sum(real(x.*conj(y)),"all");
    end

    if ~isfield(opts, 'init_yr')
        opts.init_yr = y;
    end

    if ~isfield(opts, 'init_atoms')
        opts.init_atoms = {};
        opts.init_scalars = [];
        opts.init_params = {};  
    end

    if ~isfield(opts, 'extra_arg')
        opts.extra_arg = [];
    end

    if ~isfield(opts, 'oversampling')
        opts.oversampling = 4;
    end


end