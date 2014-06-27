classdef PSplineFit < handle
    % PSplineFit learns spline-smooothed (maybe sparse) additive functions.
    %
    
    properties
        % func_points stores the 'control' points for the piecewise functions
        % learned for each input dimension. it has size F x P, where F is the
        % dimension of the input observations and P is the number of control
        % points to use for each piecewise-linear function.
        func_points
        % func_weights stores the value of the learned pwc functions at each
        % control point. 
        func_weights
        % func_consts gives constants to apply to each output dimension
        func_consts
        % cp_count gives the number of control points in each pwc function
        cp_count
        % in_dim gives the dimension of the input space
        in_dim
        % out_dim gives the dimension of the output space
        out_dim
        % out_loss is the loss function to optimize (empirically)
        out_loss
        % ord_lams weights the gradient penalty for different orders
        ord_lams
        % ord_mats gives the Tikhonov matrices for curvature regularization
        ord_mats
        % bs_order gives the B-spline basis order (only 1 or 2 are working)
        bs_order
        % lam_l1g controls a group lasso penalty on each pwl function
        lam_l1g
    end
    
    methods
        function [ self ] = PSplineFit(X, Y, opts)
            % Contructinator.
            if ~exist('opts','var')
                opts = struct();
            end
            if ~isfield(opts,'out_loss')
                opts.out_loss = @PSplineFit.loss_lsq;
            end
            if ~isfield(opts,'cp_count')
                opts.cp_count = 25;
            end
            if ~isfield(opts,'ord_lams')
                opts.ord_lams = [1e-5 1e-5 1e-5];
            end
            if ~isfield(opts,'lam_l1g')
                opts.lam_l1g = 0.0;
            end
            %if (size(Y,2) < 2)
                % Note: occasionally one runs into parts of Matlab that are
                % shockingly poorly designed. One such fault is Matlab's
                % cheerful-idiot approach to "automagically" eliding unitary
                % trailing dimensions from multi-dimensional arrays. That is,
                % arrays are not allowed to have trailing unitary dimensions,
                % and they will be removed behind your back, when you least
                % expect it. Well, actually an array can have one unitary
                % trailing dimension, but only if that trailing dimension is
                % the second dimension.
                %
            %    error('Matlab is a dimension thief, sorry.');
            %end
            do_linear = 1;
            % Set basic learning parameters
            self.in_dim = size(X,2);
            self.out_dim = size(Y,2);
            self.out_loss = opts.out_loss;
            self.cp_count = opts.cp_count;
            % Initialize the control points
            fp = PSplineFit.set_control_points(X, self.cp_count, do_linear);
            fw = randn(self.cp_count, self.in_dim, self.out_dim);
            self.func_points = fp;
            self.func_weights = fw;
            % Set up regularization stuff
            self.lam_l1g = opts.lam_l1g;
            self.ord_lams = opts.ord_lams;
            self.ord_mats = PSplineFit.get_ordmats(...
                self.ord_lams, self.func_points, do_linear);
            self.bs_order = 1;
            % Set the initial constants/biases
            self.func_consts = zeros(1, self.out_dim);
            return
        end
        
        function [ F ] = evaluate(self, X)
            % Evaluate the current set of piecewise-linear functions. The X
            % given to evaluate can be in either pwl-sparse or natural form. 
            X = self.get_sparse_obs(X);
            W = PSplineFit.get_sparse_wts(self.func_weights);
            F = bsxfun(@plus, (X * W), self.func_consts);
            return
        end
        
        function [ Xsp ] = get_sparse_obs(self, X, order)
            % Convert the observations in X to pwl-sparse form.
            %
            % In pwl-sparse form, each dimension (i.e. column) of X is encoded
            % as convex combinations of adjacent "control points", with the
            % locations of control points set independently for each dimension.
            % This encoding permits very efficient evaluation of additive
            % functions in which each additive component is piecewise linear.
            % When X is already in pwl-sparse form, we just give it back.
            %
            if ~exist('order','var')
                order = self.bs_order;
            end
            if (size(X,2) == (self.in_dim * self.cp_count))
                % X is already in sparse form, so just give it back.
                Xsp = X;
                return
            end
            if (size(X,2) ~= self.in_dim)
                % X doesn't have the right number of features, which is bad.
                error('get_sparse_obs(): X has wrong feature dimension.');
            end
            obs_count = size(X,1);
            func_count = self.in_dim;
            func_size = self.cp_count;
            if (order < 2)
                % Use first-order B-spline basis (i.e. piecewise-linear)
                x_row = zeros((obs_count * func_count * 2), 1);
                x_col = zeros((obs_count * func_count * 2), 1);
                x_wts = zeros((obs_count * func_count * 2), 1);
                cp_start = 1;
                for f_num=1:func_count,
                    % Compute starting index for this feature in sparse vector
                    f_idx = (f_num - 1) * self.cp_count;
                    % Get observation values for this feature
                    Xf = X(:,f_num);
                    % Get control points for this feature
                    p_f = self.func_points(f_num,:);
                    % Scan the 'pwl regions' for this pwl function
                    for cp_num=2:self.cp_count,
                        % Get current control point and its neighbors
                        cp_l = p_f(cp_num-1);
                        cp_r = p_f(cp_num);
                        % Get the set of observations just left of this cp
                        cp_idx = find((Xf > cp_l) & (Xf <= cp_r));
                        cp_vol = numel(cp_idx);
                        if (cp_vol > 0)
                            % Compute linterp weights for items in this region
                            cp_wts = (Xf(cp_idx) - cp_l) ./ (cp_r - cp_l);
                            % Get indices of left and right cps in sparse vector
                            l_idx = (f_idx + (cp_num - 1)) * ones(cp_vol,1);
                            r_idx = (f_idx + cp_num) * ones(cp_vol,1);
                            % Record weights in sparse representation matrix
                            cp_end = cp_start + ((cp_vol*2) - 1);
                            x_row(cp_start:cp_end) = [cp_idx; cp_idx];
                            x_col(cp_start:cp_end) = [l_idx; r_idx];
                            x_wts(cp_start:cp_end) = [(1 - cp_wts); cp_wts];
                            cp_start = cp_end + 1;
                        end
                    end
                end
            else
                % Use second-order B-spline basis (i.e. piecewise-quadratic)
                x_row = zeros((obs_count * func_count * 3), 1);
                x_col = zeros((obs_count * func_count * 3), 1);
                x_wts = zeros((obs_count * func_count * 3), 1);
                cp_start = 1;
                for f_num=1:func_count,
                    % Compute starting index for this feature in sparse vector
                    f_idx = (f_num - 1) * self.cp_count;
                    % Get observation values for this feature
                    Xf = X(:,f_num);
                    % Get control points for this feature
                    p_f = self.func_points(f_num,:);
                    % Scan the B-spline basis regions
                    for cp_num=2:(self.cp_count-1),
                        % Get the relevant control points (a.k.a. knots)
                        cp_l = p_f(cp_num-1);
                        cp_c = p_f(cp_num);
                        cp_r = p_f(cp_num+1);
                        % Find observations between cp_c and cp_r.
                        cp_idx = find((Xf > cp_c) & (Xf <= cp_r));
                        cp_vol = numel(cp_idx);
                        if (cp_vol > 0)
                            % Get weights relating observations and the
                            % relevant control points / spline knots
                            [l_wts c_wts r_wts] = PSplineFit.so_bs_wts(...
                                Xf(cp_idx), cp_l, cp_c, cp_r);
                            % Get indices of left and right cps in sparse vector
                            l_idx = (f_idx + (cp_num - 1)) * ones(cp_vol,1);
                            c_idx = (f_idx + cp_num) * ones(cp_vol,1);
                            r_idx = (f_idx + (cp_num + 1)) * ones(cp_vol,1);
                            % Record weights in sparse representation matrix
                            cp_end = cp_start + ((cp_vol*3) - 1);
                            x_row(cp_start:cp_end) = [cp_idx; cp_idx; cp_idx];
                            x_col(cp_start:cp_end) = [l_idx; c_idx; r_idx];
                            x_wts(cp_start:cp_end) = [l_wts; c_wts; r_wts];
                            cp_start = cp_end + 1;
                        end
                    end
                end
                x_row(cp_start:end) = [];
                x_col(cp_start:end) = [];
                x_wts(cp_start:end) = [];
            end
            Xsp = sparse(x_row, x_col, x_wts, obs_count, func_count*func_size);
            return
        end
        
        function [ L ] = train(self, X, Y, opts)
            % Update weights using lbfgs
            X = self.get_sparse_obs(X);
            % Define loss/gradient function for use by lbfgs
            obs_count = size(X,1);
            if ~isfield(opts, 'batch_size')
                opts.batch_size = min(obs_count, 2500);
            end
            if ~isfield(opts, 'batch_iters')
                opts.batch_iters = 10;
            end
            if ~isfield(opts, 'batch_count')
                opts.batch_count = 10;
            end
            % Create a loss function to be wrapped for use by minFunc
            function [ l dldw ] = mf_joint_loss(w, x, y)
                wb = reshape(w, (size(x,2)+1), size(y,2));
                [l_curve dldw_curve] = self.loss_curve(wb(1:(end-1),:));
                [l_out dldw_out] = self.out_loss(wb, x, y);
                l = l_curve + l_out;
                dldwb = dldw_out;
                dldwb(1:(end-1),:) = dldwb(1:(end-1),:) + dldw_curve;
                dldw = dldwb(:);
                return
            end 
            % Setup options for minFunc
            mf_opts = struct();
            mf_opts.Display = 'off';
            mf_opts.Method = 'lbfgs';
            mf_opts.Corr = 20;
            mf_opts.LS = 0;
            mf_opts.LS_init = 0;
            mf_opts.MaxIter = opts.batch_iters;
            mf_opts.MaxFunEvals = 2 * opts.batch_iters;
            mf_opts.optTol = 1e-8;
            mf_opts.TolX = 1e-8;
            % Perform optimization, beginning with current weights
            W = PSplineFit.get_sparse_wts(self.func_weights);
            b = self.func_consts;
            Wb = [W; b];
            Wb = Wb(:);
            for b=1:opts.batch_count,
                if (opts.batch_size < size(X,1))
                    b_idx = randsample(obs_count, opts.batch_size);
                    Xb = X(b_idx,:);
                    Yb = Y(b_idx,:);
                    funObj = @( w ) mf_joint_loss(w, Xb, Yb);
                    Wb = minFunc(funObj, Wb, mf_opts);
                else
                    funObj = @( w ) mf_joint_loss(w, X, Y);
                    Wb = minFunc(funObj, Wb, mf_opts);
                end
            end
            Wb = reshape(Wb, (size(X,2)+1), size(Y,2));
            W = Wb(1:(end-1),:);
            b = Wb(end,:);
            self.func_weights = ...
                PSplineFit.get_dense_wts(W, self.cp_count, self.in_dim);
            self.func_consts = b;
            return
        end
        
        function [ err ] = check_grad(self, X, Y, opts)
            % Update weights using lbfgs
            X = self.get_sparse_obs(X);
            % Define loss/gradient function for use by lbfgs
            obs_count = size(X,1);
            if ~isfield(opts, 'batch_size')
                opts.batch_size = min(obs_count, 2500);
            end
            if ~isfield(opts, 'batch_iters')
                opts.batch_iters = 10;
            end
            if ~isfield(opts, 'batch_count')
                opts.batch_count = 10;
            end
            % Create a loss function to be wrapped for use by minFunc
            function [ l dldw ] = mf_joint_loss(w, x, y)
                wb = reshape(w, (size(x,2)+1), size(y,2));
                [l_curve dldw_curve] = self.loss_curve(wb(1:(end-1),:));
                [l_out dldw_out] = self.out_loss(wb, x, y);
                l = l_curve + l_out;
                dldwb = dldw_out;
                dldwb(1:(end-1),:) = dldwb(1:(end-1),:) + dldw_curve;
                dldw = dldwb(:);
                return
            end 
            % Setup options for minFunc
            mf_opts = struct();
            mf_opts.Display = 'iter';
            mf_opts.Method = 'lbfgs';
            mf_opts.Corr = 10;
            mf_opts.LS = 0;
            mf_opts.LS_init = 0;
            mf_opts.MaxIter = opts.batch_iters;
            mf_opts.MaxFunEvals = 2 * opts.batch_iters;
            mf_opts.TolX = 1e-8;
            % Perform optimization, beginning with current weights
            W = PSplineFit.get_sparse_wts(self.func_weights);
            b = self.func_consts;
            Wb = [W; b];
            Wb = Wb(:);
            order = 1;
            type = 2;
            err = 0;
            for i=1:10,
                objFun = @( w ) mf_joint_loss(w, X, Y);
                err = err + fastDerivativeCheck(objFun, Wb, order, type);
            end
            err = mean(err);
            return
        end 
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % FUNCTION FOR EFFECTING CURVATURE PENALTY ON ADDITIVE FUNCS %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ L dLdW ] = loss_curve(self, w_sparse)
            % Get the intrinsic loss and gradients with respect to the control
            % points defining each univariate spline function.
            %
            % This also permits a "soft" group lasso penalty, to permit sparse
            % additive modeling with penalized B-splines.
            %
            W = PSplineFit.get_dense_wts(w_sparse, self.cp_count, self.in_dim);
            EPS = 1e-6;
            L = 0;
            dLdW = zeros(size(W));
            % Regularizers for each output dimension are independent.
            for i=1:self.out_dim,
                % Compute the curvature loss associated with 'w_sparse'
                w_dense = squeeze(W(:,:,i))';
                w_rdg = zeros(size(w_dense));
                for j=1:size(w_dense,1),
                    w_rdg(j,:) = w_dense(j,:) * self.ord_mats(j).K;
                end
                L_rdg = 0.5 * sum(sum(w_rdg .* w_dense, 2));
                % Compute loss due to group penalty on each univariate spline
                w_l1g = sum(w_dense.^2,2);
                L_l1g = self.lam_l1g * sum(w_l1g ./ sqrt(w_l1g + EPS));
                L = L + (L_rdg + L_l1g);
                % Compute the gradients for smoothening regularizers
                dL_rdg = w_rdg;
                % Compute the gradient for soft group lasso penalty
                dL_l1g = self.lam_l1g * ...
                    (bsxfun(@rdivide, (2*w_dense), (w_l1g + EPS).^(1/2)) - ...
                    bsxfun(@times, w_dense, (w_l1g ./ ((w_l1g + EPS).^(3/2)))));
                % Combine the gradients from smoothening/group penalties
                dLdW(:,:,i) = (dL_rdg + dL_l1g)';
            end
            dLdW = PSplineFit.get_sparse_wts(dLdW);
            return
        end
        
    end % END METHODS
    
    methods (Static = true)
        
        function [ L dLdWb ] = loss_lsq( Wb, X, Y )
            % Compute loss and gradient for least-squares regression loss.
            %
            obs_count = size(X,1);
            W = Wb(1:(end-1),:);
            b = Wb(end,:);
            % Compute current least-squares loss given weights in W
            Yh = bsxfun(@plus, (X * W), b);
            R = Yh - Y;
            L = (1 / obs_count) * sum(sum(R.^2));
            if (nargout > 1)
                % Gradient of regression loss with respect to W
                dLdW = (2 / obs_count) * (X' * R);
                dLdb = (2 / obs_count) * sum(R, 1);
                % Stack up normal parameters and biases
                dLdWb = [dLdW; dLdb];
            end
            return
        end
        
        function [l_wts c_wts r_wts] = so_bs_wts(X, l_cp, c_cp, r_cp)
            % Compute second-order B-spline knot weights for the points in X,
            % given the knot locations l_cp, c_cp, and r_cp. Points in X should
            % all be located between c_cp and r_cp.
            %
            if ((min(X) < c_cp) || (max(X) > r_cp))
                error('Invalid X values for so_bs_wts().');
            end
            pos = (X - c_cp) ./ (r_cp - c_cp);
            l_wts = 0.5 * (1 - pos).^2;
            r_wts = 0.5 * pos.^2;
            c_wts = 1 - 2*abs(0.5 - pos).^2;
            wt_sum = l_wts + c_wts + r_wts;
            l_wts = l_wts ./ wt_sum;
            c_wts = c_wts ./ wt_sum;
            r_wts = r_wts ./ wt_sum;
            return
        end
        
        function [ ord_mats ] = get_ordmats(ord_lams, cp_coords, do_linear)
            % Get 'Tikhonov' regularization matrix for multi-order curvature
            if ~exist('do_linear','var')
                do_linear = 1;
            end
            cp_count = size(cp_coords,2);
            D = zeros(cp_count, cp_count);
            D(1,1) = 1;
            for i=2:cp_count,
                D(i,i) = 1;
                D(i,i-1) = -1;
            end
            feat_count = size(cp_coords,1);
            ord_mats = struct();
            for f=1:feat_count,
                K = ord_lams(1) * eye(size(D));
                if (do_linear == -1) % (do_linear == 0)
                    % Rescale differences, to account for variable spacing
                    % between control points (a.k.a. knots). (NOT WORKING YET)
                    for i=2:cp_count,
                        scale = cp_coords(f,i) - cp_coords(f,i-1);
                        D(i,i) = D(i,i) / scale;
                        D(i,i-1) = D(i,i-1) / scale;
                    end
                end
                for i=2:numel(ord_lams),
                    C = eye(size(D));
                    for j=2:i,
                        C = C' * D;
                    end
                    for j=1:size(C,1),
                        if (sum(C(j,:) ~= 0) < i)
                            C(j,:) = 0;
                        end
                    end
                    K = K + (ord_lams(i) * (C'*C));
                end
                ord_mats(f).K = K;
            end
            return
        end
        
        function [ Wsp ] = get_sparse_wts(W)
            % Put weights into 'sparse' form (i.e. reshape to flat vector)
            dim_1 = size(W,1);
            dim_2 = size(W,2);
            dim_3 = size(W,3);
            Wsp = zeros((dim_1*dim_2), dim_3);
            for i=1:dim_3,
                Wsp(:,i) = reshape(W(:,:,i),(dim_1*dim_2),1);
            end
            %Wsp = sparse(Wsp);
            return
        end
        
        function [ W ] = get_dense_wts(Wsp, dim_1, dim_2)
            % Put weights into 'dense' form (i.e. reshape from flat vector)
            dim_3 = size(Wsp,2);
            W = zeros(dim_1, dim_2, dim_3);
            for i=1:dim_3,
                W(:,:,i) = reshape(Wsp(:,i), dim_1, dim_2);
            end
            %W = full(W);
            return
        end
        
        function [ f_points ] = set_control_points(X, cp_count, do_linear)
            % Get a set of initial control points along each dimension of X
            if ~exist('do_linear','var')
                do_linear = 1;
            end
            obs_dim = size(X,2);
            % Select a set of control points for each feature. Pairs of very
            % distant control points are set at each 'end' of the pwl, to keep
            % other computations simple. The data-dependent control points, of
            % which there are cp_count, are set either using data quantiles, or
            % with a fixed step size over the input range.
            f_points = zeros(obs_dim, cp_count);
            for d=1:obs_dim,
                Xd = X(:,d);
                if (do_linear == 1)
                   pts = linspace(min(Xd),max(Xd),(cp_count-4));
                else
                   pts = quantile(Xd,linspace(0,1,(cp_count-4)));
                end
                f_points(d,3:(end-2)) = pts;
                f_points(d,1) = -1e6;
                f_points(d,2) = -1e6 + 1;
                f_points(d,end) = 1e6;
                f_points(d,(end-1)) = 1e6 - 1;
            end
            return
        end
        
    end
   
end % END CLASSDEF

        