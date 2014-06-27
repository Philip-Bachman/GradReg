classdef TreeBooster < handle
    % A simple class for boosting short trees for real-valued observations.
    % Trees are currently learned greedily, and with a full set of leaves. The
    % constructor expects a set of initial observations X, (possibly weighted)
    % classes Y, and an options structure.
    %
    % Accepted options:
    %   nu: shrinkage/regularization term for boosting
    %   max_depth: The depth to which each tree will be extended. A max depth
    %                   of 1 corresponds to boosting stumps. All leaves at
    %                   each depth will be split. (i.e. trees are full)
    %
    
    properties
        % nu is the "shrinkage" rate used for "soft" boosting
        nu
        % loss_func determines the type of loss minimized by the trees
        loss_func
        % trees is a cell array of the trees making up this learner
        trees
        % flat_trees stores "flat" representations of each tree, for fast eval
        flat_trees
        % max_depth gives the depth to which each tree will be grown
        max_depth
    end
    
    methods
        function [ self ] = TreeBooster(X, Y, max_depth, nu)
            % Init with a constant tree
            self.loss_func = @TreeBooster.loss_lsq;
            self.flat_trees = {};
            self.nu = 1.0;
            self.max_depth = 0;
            self.extend(X,Y);
            self.nu = nu;
            self.max_depth = max_depth;
            return
        end
        
        function [ id_list ] = get_ids(self, X, tree_id)
            % Do fast traversal of the 'flattened' tree described by
            % self.flat_trees{tree_id}, to get the ids of leaves into which the
            % observations in X are sorted by the tree.
            %
            FT = self.flat_trees{tree_id};
            %obs_count = size(X,1);
            %id_list = zeros(obs_count,1);
            %for i=1:obs_count,
            %   nid = 1;
            %   x = X(i,:);
            %   while (FT(nid,3) ~= 0)
            %       if (x(FT(nid,1)) <= FT(nid,2))
            %           nid = FT(nid,3);
            %       else
            %           nid = FT(nid,4);
            %       end
            %   end
            %   id_list(i) = nid;
            %end
            id_list = fast_tree_ids(X, FT, self.max_depth+1);
            return
        end
        
        function [ weight_list ] = get_weights(self, X, tree_id)
            % Get current weights for the observations in X, as given by their
            % placement into the leaves of tree tree_id.
            %
            obs_count = size(X,1);
            FT = self.flat_trees{tree_id};
            id_list = self.get_ids(X, tree_id);
            weight_list = FT(id_list,6);
            return
        end
        
        function [ F ] = evaluate(self, X)
            % Evaluate the prediction made for each observation in X by the
            % trees from which this learner is composed.
            %
            % Parameters:
            %   X: input observations
            %
            % Output:
            %   F: the predictions for X given self.flat_trees
            %
            F = zeros(size(X,1),1);
            for t=1:length(self.flat_trees),
                weights = self.get_weights(X,t);
                F = F + weights;
            end
            return
        end
        
        function [ L ] = multi_extend(self, X, Y, iters)
            % Run extend multiple times using the training points in X/Y.
            %
            fprintf('Boosting %d rounds:',iters);
            flag = 0;
            F = zeros(size(X,1),1);
            for i=1:iters,
                if (mod(i,floor(iters/50)) == 0)
                    fprintf('.');
                end
                if (i == iters)
                    flag = 1;
                end
                L = self.extend(X,Y,flag);
                F = F + self.get_weights(X, length(self.flat_trees));
            end
            fprintf('\n');
            return
        end 
        
        function [ L ] = extend(self, X, Y, exit_eval, Fi)
            % Extend the current set of trees, based on the observations in X
            % and the loss/grad function loss_func. Return the post-update loss
            if ~exist('exit_eval','var')
                exit_eval = 1;
            end
            if exist('Fi','var')
                F = Fi;
            else
                F = self.evaluate(X);
            end
            obs_count = size(X,1);
            [L dL] = self.loss_func(F, Y, 1:obs_count);
            % Iteratively split all leaves at each current tree depth, creating
            % a full binary tree of depth self.max_depth, where a stump is
            % considered as having depth 1.
            root = TreeNode();
            root.sample_idx = 1:size(X,1);
            new_leaves = {root};
            for d=1:self.max_depth,
                old_leaves = {new_leaves{1:end}};
                new_leaves = {};
                leaf_count = length(old_leaves);
                % Split each leaf spawned by the previous round of splits
                for l_num=1:leaf_count,
                    leaf = old_leaves{l_num};
                    leaf.has_children = true;
                    leaf_idx = leaf.sample_idx;
                    if (numel(leaf_idx) > 0)
                        % Greedily split this leaf
                        leaf_X = X(leaf_idx,:);
                        leaf_dL = dL(leaf_idx);
                        [split_f split_t] = ...
                            TreeBooster.find_split(leaf_X, leaf_dL);

                    else
                        % This leaf contains none of the training samples
                        split_f = 1;
                        split_t = 0;
                    end
                    % Set split info in the split leaf
                    leaf.split_feat = split_f;
                    leaf.split_val = split_t;
                    % Create right/left children, and set their split indices
                    leaf.left_child = TreeNode();
                    leaf.right_child = TreeNode();
                    l_idx = leaf_idx(X(leaf_idx,split_f) <= split_t);
                    r_idx = leaf_idx(X(leaf_idx,split_f) > split_t);
                    leaf.left_child.sample_idx = l_idx;
                    leaf.right_child.sample_idx = r_idx;
                    % Add the newly generated leaves/children to the leaf list
                    new_leaves{end+1} = leaf.left_child;
                    new_leaves{end+1} = leaf.right_child;
                end
            end
            % Set weight in each leaf of the generated tree
            Fs = ones(size(F));
            for l_num=1:length(new_leaves),
                leaf = new_leaves{l_num};
                if (numel(leaf.sample_idx) > 0)
                    % Only set weights in leaves that contain samples
                    step_func = @( f ) self.loss_func(f, Y, leaf.sample_idx);
                    weight = TreeBooster.find_step(F, Fs, step_func);
                    leaf.weight = weight * self.nu;
                else
                    leaf.weight = 0;
                end
            end
            % Append the generated tree to the set of trees from which this
            % learner is composed.
            self.trees{end+1} = root;
            self.flat_trees{end+1} = root.get_flat_repr([],1);
            if (exit_eval == 1)
                F = self.evaluate(X);
                L = self.loss_func(F, Y, 1:obs_count);
            end
            return 
        end
        
    end
    methods (Static = true)
        
        function [best_feat best_thresh] = find_split(X, dL)
            % Compute a split of the given set of values that maximizes the
            % weighted difference of means for dL
            %
            % Parameters:
            %   X: set of observations to split
            %   dL: loss gradient with respect to each observation
            % Output:
            %   best_feat: feature on which split occurred
            %   best_thresh: threshold for split
            %
            obs_dim = size(X,2);
            obs_count = size(X,1);
            best_feat = 0;
            best_thresh = 0;
            best_sum = 0;
            % Compute the best split point for each feature, tracking best
            % feat/split pair
            for f_num=1:size(X,2),
               [f_vals f_idx] = sort(X(:,f_num),'ascend');
               f_grad = dL(f_idx);
               f_sum = 0;
               f_val = 0;
               cs_l = cumsum(f_grad);
               cs_r = -cs_l + cs_l(end);
               cs_lr = abs(cs_l) + abs(cs_r);
               [cs_vals cs_idx] = sort(cs_lr,'descend');
               % For the current feature, check all possible split points, 
               % tracking best split point and its corresponding gap
               for s_num=1:obs_count,
                   idx = cs_idx(s_num);
                   if ((idx == obs_count) || (f_vals(idx) < f_vals(idx+1)))
                       f_sum = cs_vals(s_num);
                       if (idx < obs_count)                       
                           f_val = f_vals(idx) + ...
                                       (rand()*(f_vals(idx+1)-f_vals(idx)));
                       else
                           f_val = 1e10;
                       end
                       break
                   end
               end
               % Check if the best split point found for this feature is better
               % than any split point found for previously examined features
               if (f_sum > best_sum)
                   best_sum = f_sum;
                   best_feat = f_num;
                   best_thresh = f_val;
               end
            end
            % What to do if no good split was found
            if (best_feat == 0)
                best_feat = 1;
                best_thresh = 1e10;
            end
            return
        end
        
        function [ step ] = find_step(F, Fs, step_func)
            % Use Matlab unconstrained optimization to find a step length that
            % minimizes: loss_func(F + (Fs .* step))
            step_opts = optimset('MaxFunEvals',50,'TolX',1e-3,'TolFun',...
                1e-3,'Display','off','GradObj','on');
            %function [ Ls dLdS ] = step_loss_grad(s, F, Fs, loss_func)
            %    % Wrapper function
            %    [Ls dLdS] = loss_func(F + (Fs .* s));
            %    dLdS = sum(dLdS .* Fs);
            %    return
            %end
            %objFun = @( s ) step_loss_grad(s, F, Fs, step_func);
            %step = fminunc(objFun, 0, step_opts);
            [L dL] = step_func(F);
            Fd = Fs;
            if (numel(dL) ~= numel(Fs))
                % This should only occur when estimating step for learners that
                % use a "homogeneous" step direction 
                if (sum(Fs) ~= numel(Fs))
                    error('Subindex loss function only for constant steps.\n');
                end
                Fd = ones(size(dL));
            end
            if (sum(Fd.*dL) > 0)
                step = fminbnd(@( s ) step_func(F + (Fs.*s)), -25,0,step_opts);
            else
                step = fminbnd(@( s ) step_func(F + (Fs.*s)), 0,25,step_opts);
            end
            return
        end
        
        function [ L dLdF ] = loss_lsq(F, Y, idx, loss_vec)
            % Compute the loss and gradient for least squares.
            %
            % Parameters:
            %   F: function value at each observation
            %   Y: weighted classification for each observation
            %   idx: indices at which to evaluate loss and gradients
            %   loss_vec: if this is 1, return a vector of losses
            %
            % Output:
            %   L: objective value
            %   dLdF: gradient of objective with respect to values in F
            %
            if ~exist('idx','var')
                idx = 1:size(F,1);
            end
            if ~exist('loss_vec','var')
                loss_vec = 0;
            end
            F = F(idx);
            Y = Y(idx);
            L = 0.5 * (F - Y).^2;
            if (loss_vec ~= 1)
                % Compute a vector of losses
                L = sum(L) / numel(L);
            end
            if (nargout > 1)
                % Loss gradient with respect to output at each input
                dLdF = F - Y;
            end
            return
        end
        

        function [ L dLdF ] = loss_mcl2h(F, Y, idx, loss_vec)
            % Compute the loss and gradient for multiclass 1-vs-all L2 hinge.
            %
            % Parameters:
            %   F: function value at each observation
            %   Y: weighted classification for each observation
            %   idx: indices at which to evaluate loss and gradients
            %   loss_vec: if this is 1, return a vector of losses
            %
            % Output:
            %   L: objective value
            %   dLdF: gradient of objective with respect to values in F
            %
            if ~exist('idx','var')
                idx = 1:size(F,1);
            end
            if ~exist('loss_vec','var')
                loss_vec = 0;
            end
            F = F(idx);
            Y = Y(idx);
            L = 0.5 * (F - Y).^2;
            if (loss_vec ~= 1)
                % Compute a vector of losses
                L = sum(L) / numel(L);
            end
            if (nargout > 1)
                % Loss gradient with respect to output at each input
                dLdF = F - Y;
            end
            return
        end
        
    end
    

    
end

