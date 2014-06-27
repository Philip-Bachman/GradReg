classdef TreeNode < handle
% TreeNode is a simple class/structure to be used in boosted binary decision
% trees. The TreeNode class is currently suited only to use with numerical
% features and single-feature splits based on < and > comparisons.
properties
   split_feat
   split_val
   has_children
   left_child
   right_child
   sample_idx
   sample_count
   weight
end

methods
    function self = TreeNode(s_feat, s_val)
        % construct a basic TreeNode object
        if exist('s_feat','var')
            self.split_feat = s_feat;
        else
            self.split_feat = 0;
        end
        if exist('s_val','var')
            self.split_val = s_val;
        else
            self.split_val = 0;
        end
        self.has_children = false;
        self.left_child = -1;
        self.right_child = -1;
        self.sample_idx = [];
        self.sample_count = 0;
        self.weight = 0;
    end
    
    function [ flat_repr ] = get_flat_repr(self, base_repr, start_depth)
        % Get a flat representation of the subtree rooted here, for use in
        % accelerated evaluation.
        %
        % self_repr(1) = split_feat
        % self_repr(2) = split_thresh
        % self_repr(3) = left_child id (i.e. its row in flat_repr)
        % self_repr(4) = right_child id (i.e. its row in flat_repr)
        % self_repr(5) = depth of self in overall tree being flattened
        % self_repr(6) = weight
        %
        self_repr = zeros(1,6);
        self_repr(1) = self.split_feat;
        self_repr(2) = self.split_val;
        self_repr(5) = start_depth;
        self_repr(6) = self.weight;
        flat_repr = [base_repr; self_repr];
        self_id = size(flat_repr,1);
        if self.has_children,
            flat_repr(self_id,3) = size(flat_repr,1) + 1;
            flat_repr = self.left_child.get_flat_repr(flat_repr,start_depth+1);
            flat_repr(self_id,4) = size(flat_repr,1) + 1;
            flat_repr = self.right_child.get_flat_repr(flat_repr,start_depth+1);
        end
        return
    end

    function [ leaf_list ] = get_leaf(self, X)
        % find the leaf in the subtree rooted at self to which x belongs
        leaf_list = cell(size(X,1),1);
        if self.has_children,
            right_idx = X(:,self.split_feat) >= self.split_val;
            right_leaves = self.right_child.get_leaf(X(right_idx,:));
            left_leaves = self.left_child.get_leaf(X(~right_idx,:));
            leaf_list(right_idx) = right_leaves;
            leaf_list(~right_idx) = left_leaves;    
        else
            leaf_list{1:end} = self;
        end
        return
    end
    
    function [ weight_list ] = get_weight(self, X)
        % find the weight of x in the subtree rooted at self
        weight_list = zeros(size(X,1),1);
        if self.has_children,
            right_idx = X(:,self.split_feat) >= self.split_val;
            right_weights = self.right_child.get_weight(X(right_idx,:));
            left_weights = self.left_child.get_weight(X(~right_idx,:));
            weight_list(right_idx) = right_weights;
            weight_list(~right_idx) = left_weights;    
        else
            weight_list(:) = self.weight;
        end
        return
    end
    
    function add_sample(self, s_idx)
        % add the given sample index to this self's sample index list
        self.sample_idx(self.sample_count+1) = s_idx;
        self.sample_count = self.sample_count + 1;
        return
    end
    
end % methods

end % classdef

    
    
