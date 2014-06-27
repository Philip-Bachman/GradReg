classdef ImgMatEst < handle
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
        
        function [ self ] = ImgMatEst( f_fun, o_dim, f_dim )
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
            self.block_size = 100;
            self.use_strict_len = 0;
            return
        end
        
        function [ K ] = trans_mat(self, X, trans_len)
            % Estimate a Tikhonov regularization matrix that discourages change
            % in a function with respect to small image translations.
            %
            % For all images (stored as rows of X), the constructed regularizer
            % considers small translations along the +/- x/y axes as well as
            % translations along the "main diagonals". The constructed
            % regularizer also considers clockwise and counter-clockwise
            % rotations of all images in X.
            %
            % Parameters:
            %   X: images (in rows of X) to wiggle a bit
            %   trans_len: fixed length of translations to consider (in px)
            %
            if ~exist('trans_len','var')
                trans_len = 0.5;
            end
            obs_count = size(X,1);
            Xf = self.feat_func(X);
            K = zeros(self.feat_dim,self.feat_dim);
            fprintf('Computing 8 translations:\n');
            % Do translations
            for i=1:8,
                fprintf('  %d: ', i);
                Xt = ImgMatEst.translate_ims(X, trans_len, i);
                Xtf = self.feat_func(Xt);
                X_fd = Xf - Xtf;
                Kb = X_fd' * X_fd;
                K = K + (Kb ./ obs_count);
            end
            K = (K + K') ./ 2;
            fprintf('\n');
            return
        end
        
        function [ K ] = rot_mat(self, X, rot_rads)
            % Estimate a Tikhonov regularization matrix that discourages change
            % in a function with respect to small image rotations.
            %
            % For all images (stored as rows of X), the constructed regularizer
            % considers small translations along the +/- x/y axes as well as
            % translations along the "main diagonals". The constructed
            % regularizer also considers clockwise and counter-clockwise
            % rotations of all images in X.
            %
            % Parameters:
            %   X: images (in rows of X) to wiggle a bit
            %   rot_len: fixed length of translations to consider (in px)
            %
            if ~exist('rot_rads','var')
                rot_rads = pi / 16;
            end
            obs_count = size(X,1);
            Xf = self.feat_func(X);
            K = zeros(self.feat_dim,self.feat_dim);
            fprintf('Computing 2 rotations:\n');
            % Do rotations
            for i=1:2,
                fprintf('  %d: ', i);
                Xt = ImgMatEst.rotate_ims(X, (rot_rads * sign(i - 1.5)));
                Xtf = self.feat_func(Xt);
                X_fd = Xf - Xtf;
                Kb = X_fd' * X_fd;
                K = K + (Kb ./ obs_count);
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
                [Xl Xr fd_lens] = ImgMatEst.sample_fd_endpoints(...
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
                [Xl Xr fd_lens] = ImgMatEst.sample_fd_endpoints(...
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
        
        function [ Xt ] = translate_ims(X, trans_len, trans_dir)
            % Translate images in the rows of X by the given amount, along the
            % given axis. Axes include +/- x/y and the "main diagonals", with
            % the axis ID numbers increasing clockwise, and +y is 1.
            %
            if (mod(trans_dir, 2) == 0)
                trans_len = sqrt(trans_len^2 / 2);
            end
            switch trans_dir
                case 1
                    trans_vals = [0 trans_len 1];
                case 2
                    trans_vals = [trans_len trans_len 1];
                case 3
                    trans_vals = [trans_len 0 1];
                case 4
                    trans_vals = [trans_len -trans_len 1];
                case 5
                    trans_vals = [0 -trans_len 1];
                case 6
                    trans_vals = [-trans_len -trans_len 1];
                case 7
                    trans_vals = [-trans_len 0 1];
                case 8
                    trans_vals = [-trans_len trans_len 1];
                otherwise
                    error('Unrecognized translation direction.');
            end
            trans_mat = [1 0 0; 0 1 0; 0 0 1];
            trans_mat(3,:) = trans_vals(:);
            tform = maketform('affine',trans_mat);
            im_dim = round(sqrt(size(X,2)));
            Xt = zeros(size(X));
            fprintf('Translating');
            count_count = round(size(X,1) / 40);
            for i=1:size(X,1),
                if (mod(i, count_count) == 0)
                    fprintf('.');
                end
                im = reshape(X(i,:),im_dim,im_dim);
                im = imtransform(im, tform, ...
                    'XData', [1 im_dim], 'YData', [1 im_dim]);
                Xt(i,:) = reshape(im, 1, (im_dim*im_dim));
            end
            fprintf('\n');
            return
        end
        
        function [ Xt ] = rotate_ims(X, rot_rads)
            % Rotate images in the rows of X by the given amount.
            %
            im_dim = round(sqrt(size(X,2)));
            Xt = zeros(size(X));
            fprintf('Rotating');
            count_count = round(size(X,1) / 40);
            for i=1:size(X,1),
                if (mod(i, count_count) == 0)
                    fprintf('.');
                end
                im = reshape(X(i,:),im_dim,im_dim);
                im = imrotate(im, rot_rads, 'bilinear', 'crop');
                Xt(i,:) = reshape(im, 1, (im_dim*im_dim));
            end
            fprintf('\n');
            return
        end
            
    end % END STATIC METHODS
    
end







%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%
