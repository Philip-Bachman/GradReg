classdef RegMatEst < handle
    % Basic class for managing a Tikhonov regularization matrix estimator.
    %
    
    properties
        % feat_func describes the collection of feature extractors that will be
        % treated as basis functions whose coefficients are to be regularized.
        feat_func
        % feat_dim gives the number of dimensions in outputs of feat_func
        feat_dim
        % obs_dim gives the number of dimensions in inputs to feat_func
        obs_dim
        % block_size gives the number of samples in each sample block
        block_size
        % use_strict_len
        use_strict_len
    end
    
    methods
        
        function [ self ] = RegMatEst( f_fun, o_dim, f_dim )
            % Set the "fixed" properties for this Tikhonov matrix estimator.
            % 
            % f_fun should be a function capable of extracting a full set of
            % features from any point in some input space (which we will assume
            % to be the real numbers for now).
            %
            % f_dim should give the dimension of the output of f_fun.
            %
            % Let o_dim be the dimension of inputs to f_fun... When given a
            % matrix of size (obs_count x o_dim), f_fun should return a matrix
            % of size (obs_count x f_dim).
            %
            self.feat_func = f_fun;
            self.feat_dim = f_dim;
            self.obs_dim = o_dim;
            self.block_size = 1000;
            self.use_strict_len = 0;
            return
        end
        
        function [ K ] = estimate_func(self, obs_sampler, samp_count)
            % Estimate a Tikhonov regularization matrix using sampled data.
            % This function estimates a regularizer that penalizes the standard
            % L2 functional norm.
            %
            % Data is sampled using obs_sampler, which takes as input a number
            % of samples to draw from some distribution over the input domain 
            % of self.feat_func and returns a matrix X containing the sampled
            % observations and a vector w containing the 'sample weight' for
            % each sampled observation. (weights are for importance sampling).
            %
            K = zeros(self.feat_dim,self.feat_dim);
            block_count = ceil(samp_count / self.block_size);
            samp_count = block_count * self.block_size;
            fprintf('Sampling in %d blocks: ',block_count);
            for i=1:block_count,
                % Draw a block of samples from the observation space
                [X w] = obs_sampler(self.block_size);
                % Transform sampled points into the feature space
                Xf = self.feat_func(X);
                % Compute sample-based regularizer for this block
                Kb = Xf' * bsxfun(@times, Xf, w);
                % Add to the overall sample-based regularizer
                K = K + Kb ./ samp_count;
                % Display progress indicator
                if (mod(i, floor(block_count/50)) == 0)
                    fprintf('.');
                end
            end
            K = (K + K') ./ 2;
            fprintf('\n');
            return
        end
        
        function [ K ] = estimate_grad(self, ...
                obs_sampler, grad_len, samp_count, bias)
            % Estimate a Tikhonov regularization matrix using sampled data.
            % This function estimates a regularizer that penalizes a functional
            % norm based on finite-differences gradient approximations.
            %
            % Data is sampled using obs_sampler, which takes as input a number
            % of samples to draw from some distribution over the input domain 
            % of self.feat_func and returns a matrix X containing the sampled
            % observations and a vector w containing the 'sample weight' for
            % each sampled observation. (weights are for importance sampling).
            %
            % For points in sample matrices X, pairs of endpoints bisected by
            % the rows of X are then sampled to produce pairs of points suited
            % to centered finite-differences approximations of the gradient at
            % the points originally sampled in X. Length scales of the FD
            % approximations are controlled by grad_len.
            %
            if ~exist('bias','var')
                X = obs_sampler(50);
                bias = eye(size(X,2));
            end
            K = zeros(self.feat_dim,self.feat_dim);
            block_count = ceil(samp_count / self.block_size);
            samp_count = block_count * self.block_size;
            fprintf('Sampling in %d blocks: ',block_count);
            for i=1:block_count,
                % Draw a block of samples from the observation space
                [X w] = obs_sampler(self.block_size);
                % Sample centered FD approximation endpoints based on X
                [Xl Xr fd_lens] = RegMatEst.sample_fd_endpoints(...
                    X, grad_len, bias, self.use_strict_len);
                % Transform FD endpoints into the feature space
                Xl_f = self.feat_func(Xl);
                Xr_f = self.feat_func(Xr);
                % Compute sample-based regularizer for this block (the
                % derivation of this is described in the paper).
                X_diff = Xl_f - Xr_f;
                w = w ./ (fd_lens.^2);
                Kb = X_diff' * bsxfun(@times, X_diff, w);
                % Add to the overall sample-based regularizer
                K = K + (Kb ./ samp_count);
                % Display progress indicator
                if (mod(i, floor(block_count/50)) == 0)
                    fprintf('.');
                end
            end
            K = (K + K') ./ 2;
            fprintf('\n');
            return
        end
        
        function [ K ] = estimate_hess(self, ...
                obs_sampler, grad_len, samp_count, bias)
            % Estimate a Tikhonov regularization matrix using sampled data.
            % This function estimates a regularizer that penalizes a functional
            % norm based on finite-differences Hessian approximations.
            %
            % Data is sampled using obs_sampler, which takes as input a number
            % of samples to draw from some distribution over the input domain 
            % of self.feat_func and returns a matrix X containing the sampled
            % observations and a vector w containing the 'sample weight' for
            % each sampled observation. (weights are for importance sampling).
            %
            % For points in sample matrices X, pairs of endpoints bisected by
            % the rows of X are then sampled to produce pairs of points suited
            % to centered finite-differences approximations of the Hessian at
            % the points originally sampled in X. Length scales of the FD
            % approximations are controlled by grad_len.
            %
            if ~exist('bias','var')
                Xc = obs_sampler(50);
                bias = eye(size(Xc,2));
            end
            K = zeros(self.feat_dim,self.feat_dim);
            block_count = ceil(samp_count / self.block_size);
            samp_count = block_count * self.block_size;
            fprintf('Sampling in %d blocks: ',block_count);
            for i=1:block_count,
                % Draw a block of samples from the observation space
                [Xc w] = obs_sampler(self.block_size);
                % Sample centered FD approximation endpoints based on X
                [Xl Xr fd_lens] = RegMatEst.sample_fd_endpoints(...
                    Xc, grad_len, bias, self.use_strict_len);
                % Transform FD endpoints into the feature space
                Xc_f = self.feat_func(Xc);
                Xl_f = self.feat_func(Xl);
                Xr_f = self.feat_func(Xr);
                w = w ./ (fd_lens.^4);
                % Compute sample-based regularizer for this block (the
                % derivation of this is described in the paper).
                X_diff = Xl_f + Xr_f - (2 * Xc_f);
                Kb = X_diff' * bsxfun(@times, X_diff, w);
                % Add to the overall sample-based regularizer
                K = K + (Kb ./ samp_count);
                % Display progress indicator
                if (mod(i, floor(block_count/50)) == 0)
                    fprintf('.');
                end
            end
            K = (K + K') ./ 2;
            fprintf('\n');
            return
        end
        
        function [ K_all ] = estimate_arbo(self, ...
                obs_sampler, orders, grad_len, samp_count, B)
            % Estimate a Tikhonov regularization matrix using sampled data.
            % This function estimates a regularizer that penalizes a functional
            % norm based on finite-differences gradient approximations.
            %
            % Data is sampled using obs_sampler, which takes as input a number
            % of samples to draw from some distribution over the input domain 
            % of self.feat_func and returns a matrix X containing the sampled
            % observations and a vector w containing the 'sample weight' for
            % each sampled observation. (weights are for importance sampling).
            %
            %
            if ~exist('B','var')
                X = obs_sampler(50);
                B = eye(size(X,2));
            end
            orders = sort(unique(ceil(orders)),'ascend');
            if (min(orders) < 1)
                error('Use "estimate_func" for 0th order regularizer.');
            end
            K_all = cell(1,numel(orders));
            for i=1:numel(orders),
                K_all{i} = zeros(self.feat_dim);
            end
            chain_len = max(orders)+1;
            block_count = ceil(samp_count / self.block_size);
            samp_count = block_count * self.block_size;
            fprintf('Sampling in %d blocks: ',block_count);
            for i=1:block_count,
                % Draw a block of samples from the observation space
                [Xc w] = obs_sampler(self.block_size);
                % Sample FD chains of length adequate for the maximum order FD
                % we will be estimating.
                [X_f X_b fd_lens] = RegMatEst.sample_fd_chain(...
                    Xc, chain_len, grad_len, B, self.use_strict_len);
                % Transform FD chains into the feature space
                F_f = cell(1,length(X_f));
                %F_b = cell(1,length(X_f));
                for j=1:length(X_f),
                    F_f{j} = self.feat_func(X_f{j});
                    %F_b{j} = self.feat_func(X_b{j});
                end
                % For each desired order, update the approximate regularizer
                for j=1:numel(orders),
                    ord = orders(j);
                    F_fd = RegMatEst.fd_diffs(F_f, ord);
                    w = w ./ (fd_lens.^(2*ord));
                    K_ord = F_fd' * bsxfun(@times, F_fd, w);
                    K_all{j} = K_all{j} + ((K_ord + K_ord') ./ 2);
                end
                % Display progress indicator
                if (mod(i, floor(block_count/50)) == 0)
                    fprintf('.');
                end
            end
            max_eig = -1;
            for i=1:numel(orders),
                K = K_all{i};
                eig_i = max(eig(K));
                if (eig_i > max_eig)
                    max_eig = eig_i;
                end
            end
            for i=1:numel(orders),
                K = K_all{i};
                K = double((K + K') ./ 2);
                %K_all{i} = K ./ max(eig(K));
                %K_all{i} = K ./ max_eig;
            end
            fprintf('\n');
            return
        end
        
        function [ K_multi used_counts ] = estimate_arbo_multi(self, ...
                obs_sampler, orders, grad_len, samp_counts, B)
            % Estimate gradient regularizer over multiple sample counts.
            %
            if ~exist('B','var')
                X = obs_sampler(50);
                B = eye(size(X,2));
            end
            orders = sort(unique(ceil(orders)),'ascend');
            samp_counts = sort(samp_counts,'ascend');
            K_all = cell(1,numel(orders));
            for i=1:numel(orders),
                K_all{i} = zeros(self.feat_dim);
            end
            K_multi = cell(1,numel(samp_counts));
            chain_len = max(orders)+1;
            block_count = ceil(max(samp_counts) / self.block_size);
            used_counts = zeros(1,numel(samp_counts));
            total_samps = 0;
            cur_idx = 1;
            fprintf('Sampling in %d blocks: ',block_count);
            for i=1:(2*block_count),
                % If we've passed a desired sample count, record the set of
                % partially approximated regularization matrices
                if (total_samps > samp_counts(cur_idx))
                    K_cur = cell(1,numel(orders));
                    for j=1:numel(orders),
                        K_cur{j} = K_all{j} ./ (total_samps / self.block_size);
                    end
                    K_multi{cur_idx} = K_cur;
                    used_counts(cur_idx) = total_samps;
                    cur_idx = cur_idx + 1;
                    if (cur_idx > numel(samp_counts))
                        break;
                    end
                end
                % Draw a block of samples from the observation space
                [Xc w] = obs_sampler(self.block_size);
                % Sample FD chains of length adequate for the maximum order FD
                % we will be estimating.
                [X_f X_b fd_lens] = RegMatEst.sample_fd_chain(...
                    Xc, chain_len, grad_len, B, self.use_strict_len);
                % Transform FD chains into the feature space
                F_f = cell(1,length(X_f));
                for j=1:length(X_f),
                    F_f{j} = self.feat_func(X_f{j});
                end
                % For each desired order, update the approximate regularizer
                for j=1:numel(orders),
                    ord = orders(j);
                    F_fd = RegMatEst.fd_diffs(F_f, ord);
                    w = w ./ (fd_lens.^(2*ord));
                    K_ord = F_fd' * bsxfun(@times, F_fd, w);
                    K_ord = ((K_ord + K_ord') ./ (2*self.block_size));
                    K_all{j} = K_all{j} + ((K_ord + K_ord') ./ 2);
                end
                % Track the nuber of samples processed thus far
                total_samps = total_samps + self.block_size;
                % Display progress indicator
                if (mod(i, floor(block_count/50)) == 0)
                    fprintf('.');
                end
            end
            fprintf('\n');
            return
        end
        
    end % END INSTANCE METHODS
    
    methods (Static = true)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % METHODS FOR CONSTRUCTING SAMPLERS BASED ON SOME DATA %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ sample_func ] = obs_sampler( X )
            % Construct a sampler designed for drawing sample observations from
            % a distribution approximating the source distribution of the data
            % in X. Do this simply by sampling points from X.
            %
            function [ Xs w ] = do_sampling( Xo, samp_count )
                idx = randi(size(Xo,1), samp_count, 1);
                Xs = Xo(idx,:);
                w = ones(samp_count, 1);
                return
            end
            % Function handle capable of sampling from gm_obj
            sample_func = @( samp_count ) do_sampling(X, samp_count);
            return
        end
        
        function [ sample_func ] = obs_sampler_fuzzy( X, fuzz_len )
            % Construct a sampler designed for drawing sample observations from
            % a distribution approximating the source distribution of the data
            % in X. Do this simply by sampling points from X and then adding
            % Gaussian noise with standard deviation fuzz_len.
            %
            function [ Xs w ] = do_sampling( Xo, samp_count, fuzz_std )
                idx = randi(size(Xo,1), samp_count, 1);
                Xd = randn(samp_count,size(Xo,2));
                Xd = bsxfun(@rdivide, Xd, max(sqrt(sum(Xd.^2,2)),1e-8));
                Xd_l = fuzz_std * randn(samp_count,1);
                Xd_l = Xd_l .* ((rand(samp_count,1) > 0.95) + 1e-3);
                Xd = bsxfun(@times, Xd, Xd_l);
                Xs = Xo(idx,:) + Xd;
                w = ones(samp_count, 1);
                return
            end
            % Function handle capable of sampling from gm_obj
            sample_func = @( samp_count ) do_sampling(X, samp_count, fuzz_len);
            return
        end
        
        function [ sample_func ] = obs_sampler_inv( X, fuzz_len, k_width )
            % Construct a sampler designed for drawing sample observations from
            % a distribution approximating the source distribution of the data
            % in X. Do this simply by sampling points from X and then adding
            % Gaussian noise with standard deviation fuzz_len.
            %
            % The returned sample weights are inversely proportional to an
            % approximation of the density of X computed by a kernel-weighted
            % estimation.
            %
            Nx = sum(X.*X,2);
            Dx = bsxfun(@minus, bsxfun(@plus,Nx,Nx'), 2*(X*X'));
            Kx = exp(-(Dx ./ (k_width.^2)));
            Wx = 1 ./ sum(Kx,2);
            Wx = Wx.^2;
            Wx = Wx ./ sum(Wx);
            function [ Xs w ] = do_sampling( Xo, Wo, samp_count, fuzz_std )
                idx = randi(size(Xo,1), samp_count, 1);
                Xd = randn(samp_count,size(Xo,2));
                Xd = bsxfun(@rdivide, Xd, max(sqrt(sum(Xd.^2,2)),1e-8));
                Xd_l = fuzz_std * randn(samp_count,1);
                Xd_l = Xd_l .* ((rand(samp_count,1) > 0.95) + 1e-3);
                Xd = bsxfun(@times, Xd, Xd_l);
                Xs = Xo(idx,:) + Xd;
                w = Wo(idx);
                return
            end
            % Function handle capable of sampling from gm_obj
            sample_func = @( samp_count ) do_sampling(X,Wx,samp_count,fuzz_len);
            return
        end
        
        function [ sample_func ] = box_sampler( X, fuzz_len )
            % Construct a sampler to sample uniformly from the smallest
            % axis-aligned hyperrectangle enclosing the points in X.
            %
            X_min = min(X);
            X_range = max(X) - X_min;
            function [ Xs w ] = do_sampling( samp_count, Xr, Xm, fuzz_std )
                % Sample from the desired box
                Xs = bsxfun(@plus,bsxfun(@times,...
                    rand(samp_count,numel(Xm)), Xr), Xm);
                % Sample blur/displacement vectors for the points
                Xd = randn(samp_count,size(Xs,2));
                Xd = bsxfun(@rdivide, Xd, max(sqrt(sum(Xd.^2,2)),1e-8));
                Xd_l = fuzz_std * randn(samp_count,1);
                Xd = bsxfun(@times, Xd, Xd_l);
                Xs = Xs + Xd;
                w = ones(samp_count, 1);
                return
            end
            sample_func = @( obs_count ) ...
                do_sampling(obs_count,X_range,X_min,fuzz_len);
            return
        end
        
        function [ sample_func ] = compound_sampler(smplr_1, smplr_2, prob)
            % Draw samples from a pair of samplers, using a sample from smplr_1
            % with probability prob and a sample from smplr_2 otherwise.
            %
            function [ Xs w ] = do_sampling( samp_count, s1, s2, p1 )
                s1_count = sum(rand(samp_count,1) < p1);
                s2_count = samp_count - s1_count;
                [samples_1 weights_1] = s1(s1_count);
                [samples_2 weights_2] = s2(s2_count);
                Xs = [samples_1; samples_2];
                w = [weights_1; weights_2];
                return
            end
            sample_func = @( obs_count ) ...
                do_sampling(obs_count, smplr_1, smplr_2, prob);
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % METHODS FOR FD RELATED STUFF %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [Xl Xr fd_lens] = sample_fd_endpoints(...
                Xc, grad_len, bias, strict_len)
            % Sample a pair of centered FD endpoints for each row of X. Offset
            % each endpoint from its corresponding row in X by grad_len. If
            % given, transform the directions by some bias (in matrix form).
            %
            % Sample offsets uniformly at random from the surface of a
            % hypersphere of radius grad_len, embedded in a space with
            % dimensionality determined by X.
            %
            if ~exist('bias','var')
                bias = eye(size(Xc,2));
            end
            if ~exist('strict_len','var')
                strict_len = 0;
            end
            % Sample offset directions, and transform by the bias matrix
            Xd = randn(size(Xc));
            Xd = Xd * bias;
            Xd = bsxfun(@rdivide, Xd, max(sqrt(sum(Xd.^2,2)),1e-8));
            if (strict_len ~= 1)
                % Sample length scales from a scaled abs(normal) distribution
                fd_lens = grad_len * abs(randn(size(Xd,1),1));
                fd_lens = max(fd_lens,(grad_len/4));
                %% Sample length scales from a lognormal distribution               
                %m = grad_len;
                %v = grad_len / 2;
                %mu = log((m^2)/sqrt(v+m^2));
                %sigma = sqrt(log(v/(m^2)+1));
                %grad_lens = lognrnd(mu,sigma,size(Xd,1),1);
                %grad_lens = max(grad_lens, (grad_len/4));
            else
                fd_lens = grad_len * ones(size(Xd,1),1);
            end
            % Rescale sampled directions
            Xd = bsxfun(@times, Xd, fd_lens);
            % Construct endpoints by combining rows of X with offsets
            Xl = Xc - Xd;
            Xr = Xc + Xd;
            return
        end
        
        function [ X_f X_b fd_lens ] = ...
                sample_fd_chain(Xc, chain_len, fd_len, bias, strict_len)
            % Sample a chain of points for forward and backward FD estimates of 
            % directional higher-order derivatives emanating from points in Xc.
            %
            % Sample chain directions from a uniform distribution over the
            % surface of a hypersphere of dimension size(X,2). Then, rescale
            % these directions based on grad_len. If given, apply the "biasing"
            % transform (a matrix) to the directions prior to setting lengths.
            %
            if ~exist('bias','var')
                bias = eye(size(Xc,2));
            end
            if ~exist('strict_len','var')
                strict_len = 0;
            end
            % Sample offset directions, and transform by the bias matrix
            Xd = randn(size(Xc));
            Xd = Xd * bias;
            Xd = bsxfun(@rdivide, Xd, max(sqrt(sum(Xd.^2,2)),1e-8));
            MIN_LEN = 0.1;
            if (strict_len ~= 1)
                % Sample length scales from a scaled abs(normal) distribution
                fd_lens = zeros(size(Xd,1),1);
                for i=1:4,
                    rnd_lens = fd_len * abs(randn(size(fd_lens)));
                    fd_lens(fd_lens < MIN_LEN) = rnd_lens(fd_lens < MIN_LEN);
                end
                fd_lens = max(fd_lens,MIN_LEN);
                % Sample length scales from a lognormal distribution
                %m = fd_len;
                %v = fd_len / 2;
                %mu = log((m^2)/sqrt(v+m^2));
                %sigma = sqrt(log(v/(m^2)+1));
                %fd_lens = lognrnd(mu,sigma,size(Xd,1),1);
                %fd_lens = max(fd_lens, MIN_LEN);
            else
                fd_lens = fd_len * ones(size(Xd,1),1);
            end
            % Rescale sampled directions
            Xd = bsxfun(@times, Xd, fd_lens);
            % Construct f/b chains by stepwise displacement from rows of Xc.
            X_f = cell(1,chain_len);
            X_b = cell(1,chain_len);
            for i=0:(chain_len-1),
                X_f{i+1} = Xc + (i * Xd);
                X_b{i+1} = Xc - (i * Xd);
            end
            return
        end
        
        function [ Xd ] = fd_diffs(X_all, fd_order)
            % Compute FD vectors, for FD estimates of order 'fd_order', under
            % the assumption that X_all is a cell array of len >= fd_order+1,
            % whose elements are matrices, whose corresponding rows represent
            % sequences of points in FD chains.
            %
            if ((fd_order + 1) > length(X_all))
                error('Insufficient FD chain length.');
            end
            Xd = zeros(size(X_all{1}));
            for i=0:fd_order,
                Xi = X_all{i+1};
                Xd = Xd + (((-1)^i * nchoosek(fd_order,i)) * Xi);
            end
            return
        end
            

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % METHODS FOR COMPUTING KNN AND FD LENGTH SCALES %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ I_nn ] = knn_ind( Xte, Xtr, k, do_loo, do_dot )
            % Find indices of the knn for points in Xte measured with respect to
            % the points in Xtr.
            %
            % (do_loo == 1) => do "leave-one-out" knn, assuming Xtr == Xte
            % (do_dot == 1) => use max dot-products instead of min euclideans.
            %
            if ~exist('do_loo','var')
                do_loo = 0;
            end
            if ~exist('do_dot','var')
                do_dot = 0;
            end
            obs_count = size(Xte,1);
            I_nn = zeros(obs_count,k);
            fprintf('Computing knn:');
            for i=1:obs_count,
                if (mod(i,round(obs_count/50)) == 0)
                    fprintf('.');
                end
                if (do_dot == 1)
                    d = Xtr * Xte(i,:)';
                    if (do_loo == 1)
                        d(i) = min(d) - 1;
                    end
                    [d_srt i_srt] = sort(d,'descend');
                else
                    d = sum(bsxfun(@minus,Xtr,Xte(i,:)).^2,2);
                    if (do_loo == 1)
                        d(i) = max(d) + 1;
                    end
                    [d_srt i_srt] = sort(d,'ascend');
                end
                I_nn(i,:) = i_srt(1:k);
            end
            fprintf('\n');
            return
        end
        
        function [Inn Dnn] = compute_knn( X, k )
            % Find indices of, and distances to, the k-nearest-neighbors for
            % points in X.
            %
            obs_count = size(X,1);
            Inn = zeros(obs_count, k);
            Dnn = zeros(obs_count, k);
            fprintf('Computing nearest neighbors:');
            for i=1:obs_count,
                if (mod(i, floor(obs_count/40)) == 0)
                    fprintf('.');
                end
                x = X(i,:);
                d = sqrt(sum(bsxfun(@minus,X,x).^2,2));
                d(i) = max(d) + 1;
                [val idx] = sort(d,'ascend');
                Inn(i,:) = idx(1:k)';
                Dnn(i,:) = val(1:k)';
            end
            fprintf('\n');
            return
        end
        
        function [ grad_len ] = compute_grad_len(X, sample_count)
            % Compute a length scale for gradient/hessian regularization. 
            % Sample some observations at random and compute their nearest
            % neighbors in X. Use some function of the computed nearest
            % neighbors distances as the length scale for finite-differences.
            %
            obs_count = size(X,1);
            dists = zeros(sample_count,1);
            for i=1:sample_count,
                idx = randi(obs_count);
                x1 = X(idx,:);
                dx = sqrt(sum(bsxfun(@minus,X,x1).^2,2));
                dx(idx) = max(dx) + 1;
                dists(i) = min(dx);
            end
            grad_len = median(dists) / 2;
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % GENERAL MATRIX/LINEAR ALGEBRA STUFF %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ D ] = gaussy_divergence(M1, M0, lam)
            % Compute a divergence between PSD matrices M1 and M0. This is
            % equivalent to kl(N(0,M1) || N(0,M0)), i.e. KL divergence between
            % the Gaussian distribution implied by M1 and that implied by M0.
            %
            if ~exist('lam','var')
                lam = 1e-5;
            end
            M1 = M1 + (lam*eye(size(M1)));
            M0 = M0 + (lam*eye(size(M0)));
            d = size(M1,1);
            M0i = pinv(M0);
            D = trace(M1 * M0i) - log(det(M1 * M0i)) - d;
            return
        end
        
        function [ D ] = regmat_divergence(M1, M0, lam)
            % Compute a divergence between PSD matrices M1 and M0. This is
            % equivalent to max_{v in unit ball} |v'*M1v - v'*M0v|. This value
            % can be computed efficiently from spectrum of (M1 - M0).
            %
            if ~exist('lam','var')
                lam = 1e-5;
            end
            M1 = M1 + (lam*eye(size(M1)));
            M0 = M0 + (lam*eye(size(M0)));
            D = max(abs(eig(M0 - M1)));
            return
        end
        
        function [ Md ] = fd_block_matrix(fd_order, fd_count)
            % Compute a simple finite-differences block diagonal matrix. Each
            % block comprises a "chain" of fd_order+1 differenced points, and
            % there fd_count repretitions of such blocks on the diagonal.
            %
            Md = zeros((fd_order+1) * fd_count);
            fd_end = 0;
            for fd_num=1:fd_count,
                fd_start = fd_end + 1;
                fd_end = fd_start + fd_order;
                for fd_val=fd_start:fd_end,
                    Md(fd_val,fd_val) = -1;
                    if (fd_val < fd_end)
                        Md(fd_val,fd_val+1) = 1;
                    end
                end
            end
            return
        end
        
        function [ K_mix ] = mix_regmats(K_all, alpha, max_order)
            % Mix the multi-order collection of regularizers in K_mix.
            %
            if ~exist('max_order','var')
                max_order = length(K_all);
            end
            if (max_order > length(K_all))
                max_order = length(K_all);
            end
            K_mix = zeros(size(K_all{1}));
            for i=1:max_order,
                K_mix = K_mix + (exp(-alpha * (i - 1)) * K_all{i});
            end
            return
        end
        
        function [ K_mix ] = mix_rm_rbfish(K_all, orders, sigma)
            % Mix Lappy regularizers of different orders sort of RBFishly.
            %
            if ~exist('sigma','var')
                sigma = 1;
            end
            if (length(K_all) ~= numel(orders))
                error('Each regmat in K_all requires an order.');
            end
            K_mix = zeros(size(K_all{1}));
            for i=1:length(K_all),
                o = orders(i);
                K_mix = K_mix + ((sigma^(2*o)/(2^o * factorial(o)))*K_all{i});
            end
            
            return
        end
            
    end % END STATIC METHODS
    
end







%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%
