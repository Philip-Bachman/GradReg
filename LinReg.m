classdef LinReg < handle
    % Basic class for managing a flexible linear regressor.
    %
    % This class requires a feature transform (self.feat_func) that will be
    % applied to inputs prior to training or evaluation. If you're not
    % particularly hip, then just set self.feat_func = @( X ) X.
    %
    % This class implements four primary optimization objectives and three
    % types of regularization.
    % 
    % For classification, you can choose multinomial logistic regression or
    % 1-vs-all squared hinge loss. For regression, you can choose standard
    % least-squares or Huberized least-squares.
    %
    % For regularization, you can independently control the strength of L2
    % regularization, generalized L2 (a.k.a. Tikhonov) regularization, and a
    % soft approximation of L1 (a.k.a. Lasso) regularization. At training time,
    % to use Tikhonov regularization you will also need to define an appropriate
    % square, symmetric, positive semi-definite matrix K to use in w'*K*w.
    %
    
    properties
        % feat_func gives the transform to apply to all inputs
        feat_func
        % W holds the current coefficient estimates for each output dimension
        W
        % lam_rdg gives strength of standard L2 regularization
        lam_rdg
        % lam_las gives strength of approximate L1 regularization
        lam_las
        % lam_fun gives strength of Tikhonovy L2 regularization
        lam_fun
        % loss_func determines which loss function to optimize. A set of
        % appropriate loss functions are implemented as static methods of the
        % LinReg class. They all take the same parameters as input and produce
        % the same set of outputs, so they are effectively interchangeable.
        %
        %   For regression:
        %     LinReg.lsq_loss_grad: standard least-squares loss
        %     LinReg.hub_loss_grad: huberized least-squares loss (Google it)
        %   For classification:
        %     LinReg.mclr_loss_grad: multiclass logistic regression loss
        %     LinReg.mcl2_loss_grad: multiclass 1-vs-all squared hinge loss
        %
        loss_func
    end
    
    methods
        
        function [ self ] = LinReg(X, Y, feat_func, opts)
            % Simple constructor to initialize this learner.
            %
            if ~exist('opts','var')
                opts = struct();
            end
            if ~isfield(opts,'lam_rdg')
                opts.lam_rdg = 1e-5;
            end
            if ~isfield(opts,'lam_las')
                opts.lam_las = 0.0;
            end
            if ~isfield(opts,'lam_fun')
                opts.lam_fun = 0.0;
            end
            self.feat_func = feat_func;
            obs_dim = size(self.feat_func(X),2);
            out_dim = size(Y,2);
            self.W = zeros(obs_dim, out_dim);
            self.lam_rdg = opts.lam_rdg;
            self.lam_las = opts.lam_las;
            self.lam_fun = opts.lam_fun;
            self.loss_func = @LinReg.lsq_loss_grad;
            return
        end
        
        function [ Yh ] = evaluate(self, X)
            % Evaluate at the points in X using the current weights.
            %
            Xt = self.feat_func(X);
            Yh = Xt * self.W;
            return
        end
        
        function [ result ] = train(self, X, Y, K)
            % Use minFunc to learn a Tikhonov-regularized regression. If a
            % classification objective is currently selected for self.loss_func,
            % then Y should be encoded in a +1/-1 indicator matrix.
            %
            % Parameters:
            %   X: training observations (to be transformed by self.feat_func)
            %   Y: outputs (real-valued) for training observations
            %   K: kernelish matrix for psuedo-RKHS (Tikhonov) regularization
            % Outputs:
            %   result: training result info
            %
            Xt = self.feat_func(X);
            if ~exist('K','var')
                K = zeros(size(Xt,2));
            end
            obs_dim = size(Xt,2);
            out_dim = size(Y,2);
            % Setup options structure for minFunc
            opts_mf = struct();
            opts_mf.Display = 'iter';
            opts_mf.Method = 'lbfgs';
            opts_mf.Corr = 20;
            opts_mf.Damped = 0;
            opts_mf.useMex = 0;
            opts_mf.LS = 3;
            opts_mf.LS_init = 1;
            opts_mf.MaxIter = 500;
            opts_mf.MaxFunEvals = 2000;
            opts_mf.TolX = 1e-8;
            % Setup a loss function for use by minFunc
            el_funco = @( w ) self.loss_func(w, Xt, Y, K,...
                out_dim, self.lam_rdg, self.lam_las, self.lam_fun);
            % Run minFunc to compute optimal SVM parameters
            W_init = reshape(self.W, obs_dim*out_dim, 1);
            W_init = randn(size(W_init));
            W_min = minFunc(el_funco, W_init, opts_mf);
            self.W = reshape(W_min, obs_dim, out_dim);
            % Record result info
            [L dL L_out] = el_funco(W_min);
            result = struct();
            result.W = self.W;
            result.L = L;
            result.L_out = L_out;
            return
        end
        
        function [ acc Yh ] = test(self, X, Y)
            % Test regression accuracy of the current weights for the
            % observations in X, given target outputs in Y.
            %
            Yh = self.evaluate(X);
            acc = 1 - (var(Yh(:)-Y(:)) / var(Y(:)));
            return
        end
        
        function [ Yh acc ] = classify(self, X, Y)
            % Perform classification for the observations in X, and measure
            % classification accuracy w.r.t. to the target classes in Y.
            %
            Yh = self.evaluate(X);
            [vals c_idx] = max(Yh,[],2);
            Yh = c_idx;
            [vals c_idx] = max(Y,[],2);
            Y = c_idx;
            acc = sum(Y == Yh) / numel(Y);
            return
        end
            
    end % END INSTANCE METHODS
    
    methods (Static = true)
        
       function [ L dLdW L_out ] = lsq_loss_grad(...
               w, X, Y, K, class_count, lam_rdg, lam_las, lam_fun)
            % Compute loss and gradient for least-squares regression loss.
            %
            [obs_count, obs_dim] = size(X);
            W = reshape(w, obs_dim, class_count);
            % Merge ridge into general Tikhonov regularizer
            K = (lam_rdg * eye(size(K))) + (lam_fun * K);
            % Compute current least-squares loss given weights in W
            R = Y - (X * W);
            L_out = (1 / (2*obs_count)) * sum(sum(R.^2));
            % Compute Tikhonov loss (i.e. trace(W' * X' * X * W))
            L_tik = (1 / 2) * sum(sum(W .* (K * W)));
            % Compute L1(ish) lasso regularization loss for W
            L_las = lam_las * sum(sum( (W.^2) ./ sqrt(W.^2 + 1e-8) ));
            % Compute total loss
            L = L_out + L_tik + L_las;
            if (nargout > 1)
                % Gradient of regression loss with respect to W
                dLdW_out = -(1/obs_count) * (X' * R);
                % Gradient of Tikhonov loss with respect to W
                dLdW_tik = K * W;
                % Gradient of lasso loss with respect to W
                r = W ./ sqrt(W.^2 + 1e-8);
                dLdW_las = lam_las * ((2 * r) - (r.^3));
                % Compute the overall gradient with respect to W
                dLdW = dLdW_out + dLdW_tik + dLdW_las;
                dLdW = reshape(dLdW, (obs_dim * class_count), 1);
            end
            return
       end
       
       function [ L dLdW L_out ] = hub_loss_grad(...
               w, X, Y, K, class_count, lam_rdg, lam_las, lam_fun)
            % Compute loss and gradient for huberized regression loss.
            %
            [obs_count, obs_dim] = size(X);
            W = reshape(w, obs_dim, class_count);
            % Merge ridge into general Tikhonov regularizer
            K = (lam_rdg * eye(size(K))) + (lam_fun * K);
            % Compute current least-squares loss given weights in W
            R = Y - (X * W);
            R_abs = R .* abs(R > 1);
            R_lsq = R .* abs(R <= 1);
            L_out = ((1 / (2*obs_count)) * sum(R_lsq(:).^2)) + sum(R_abs(:));
            % Compute Tikhonov loss (i.e. trace(W' * X' * X * W))
            L_tik = (1 / 2) * sum(sum(W .* (K * W)));
            % Compute L1(ish) lasso regularization loss for W
            L_las = lam_las * sum(sum( (W.^2) ./ sqrt(W.^2 + 1e-8) ));
            % Compute total loss
            L = L_out + L_tik + L_las;
            if (nargout > 1)
                % Gradient of regression loss with respect to W
                dLdW_out = -(1/obs_count) * (X' * max(-1,min(1,R)));
                % Gradient of Tikhonov loss with respect to W
                dLdW_tik = K * W;
                % Gradient of lasso loss with respect to W
                r = W ./ sqrt(W.^2 + 1e-8);
                dLdW_las = lam_las * ((2 * r) - (r.^3));
                % Compute the overall gradient with respect to W
                dLdW = dLdW_out + dLdW_tik + dLdW_las;
                dLdW = reshape(dLdW, (obs_dim * class_count), 1);
            end
            return
       end
       
       function [ L dLdW L_cls ] = mclr_loss_grad(...
               w, X, Y, K, class_count, lam_rdg, lam_las, lam_fun)
            % Compute loss and gradient for 1-vs-all logistic loss function.
            %
            [vals Y] = max(Y,[],2);
            [obs_count, obs_dim] = size(X);
            W = reshape(w, obs_dim, class_count);
            % Merge ridge into general Tikhonov regularizer
            K = (lam_rdg * eye(size(K))) + (lam_fun * K);
            % Make a class indicator matrix using +1/-1
            Y = bsxfun(@(y1,y2) (2*(y1==y2))-1, Y, 1:class_count);
            [Y_max Y_idx] = max(Y,[],2);
            % Compute current logistic predictions given weights in W
            Yh = X * W;
            P = bsxfun(@rdivide, exp(Yh), sum(exp(Yh),2));
            % Compute classification loss (deviance)
            p_idx = sub2ind(size(P), (1:obs_count)', Y_idx);
            L_cls = (1 / obs_count) * sum(sum(-log(P(p_idx))));
            % Compute Tikhonov loss (i.e. trace(W' * X' * X * W))
            L_tik = (1 / 2) * sum(sum(W .* (K * W)));
            % Compute L1(ish) lasso regularization loss for W
            L_las = lam_las * sum(sum( (W.^2) ./ sqrt(W.^2 + 1e-8) ));
            % Compute total loss
            L = L_cls + L_tik + L_las;
            if (nargout > 1)
                % Gradient of classification loss with respect to W
                Yi = bsxfun(@eq, Y_idx, 1:class_count);
                dL_cls = P - Yi;
                dLdW_cls = (1/obs_count) * (X' * dL_cls);
                % Gradient of Tikhonov loss with respect to W
                dLdW_tik = K * W;
                % Gradient of lasso loss with respect to W
                r = W ./ sqrt(W.^2 + 1e-8);
                dLdW_las = lam_las * ((2 * r) - (r.^3));
                % Compute the overall gradient with respect to W
                dLdW = dLdW_cls + dLdW_tik + dLdW_las;
                dLdW = reshape(dLdW, (obs_dim * class_count), 1);
            end
            return
       end
       
       function [ L dLdW L_cls ] = mcl2_loss_grad(...
               w, X, Y, K, class_count, lam_rdg, lam_las, lam_fun)
            % Compute loss and gradient for 1-vs-all L2-svm loss function.
            %
            [vals Y] = max(Y,[],2);
            [obs_count, obs_dim] = size(X);
            W = reshape(w, obs_dim, class_count);
            % Merge ridge into general Tikhonov regularizer
            K = (lam_rdg * eye(size(K))) + (lam_fun * K);
            % Make a class indicator matrix using +1/-1
            Y = bsxfun(@(y1,y2) (2*(y1==y2))-1, Y, 1:class_count);
            % Compute current L2 hinge loss given weights in W
            margin = max(0, 1 - (Y .* (X * W)));
            L_cls = (1 / obs_count) * sum(sum(margin.^2));
            % Compute Tikhonov loss (i.e. trace(W' * X' * X * W))
            L_tik = (1 / 2) * sum(sum(W .* (K * W)));
            % Compute L1(ish) lasso regularization loss for W
            L_las = lam_las * sum(sum( (W.^2) ./ sqrt(W.^2 + 1e-8) ));
            % Compute total loss
            L = L_cls + L_tik + L_las;
            if (nargout > 1)
                % Gradient of classification loss with respect to W
                dLdW_cls = -((2/obs_count) * (X' * (margin .* Y)));
                % Gradient of Tikhonov loss with respect to W
                dLdW_tik = K * W;
                % Gradient of lasso loss with respect to W
                r = W ./ sqrt(W.^2 + 1e-8);
                dLdW_las = lam_las * ((2 * r) - (r.^3));
                % Compute the overall gradient with respect to W
                dLdW = dLdW_cls + dLdW_tik + dLdW_las;
                dLdW = reshape(dLdW, (obs_dim * class_count), 1);
            end
            return
       end
       
    end % END STATIC METHODS
    
end







%%%%%%%%%%%%%%
% EYE BUFFER %
%%%%%%%%%%%%%%