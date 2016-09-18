function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
	
    
    %Bias
    
    X = [ones(m,1) X];
	
	%Forward propagation
	Z2 = Theta1 * X';
	
	A2 = [ones(m, 1) sigmoid(Z2)'];	
	Z3 = Theta2 * A2';

	hypothesis = sigmoid(Z3);
	
	
    y_k = zeros(num_labels, m);
    for i = 1: m
        y_k(y(i),i) = 1;
    end

	%Unregularized cost
		
	J = 1/m * sum(sum( -y_k .* log(hypothesis) - (1 - y_k) .* log(1 - hypothesis)));

	
	%Add the regularized cost
	
	% Dropping the bias coloumns of Theta 1 and Theta 2	
	T1 = Theta1(:,2:size(Theta1,2));
	T2 = Theta2(:,2:size(Theta2,2));
	
	Reg = lambda / (2*m) * (sum(sum(T1.^2)) + sum(sum(T2.^2)));
	
	J = J + Reg;
	
	
	
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
	
	for t = 1:m
		
        %%%%%%%%%%%%%%%%%%%%%
        % Feed forward pass %
        %%%%%%%%%%%%%%%%%%%%%
        
        a_1 = X(t,:);

        z_2 = Theta1 * a_1';

        a_2 = sigmoid(z_2);
        a_2 = [1; a_2]; %Adding bias
        	
        z_3 = Theta2 * a_2;

        a_3 = sigmoid(z_3);
        

        %%%%%%%%%%%%%%%%%%%%
        % Back propagation %
        %%%%%%%%%%%%%%%%%%%%
        
        %Caculating deltas
        
        delta_3 = a_3 - y_k(:,t);
        
        z_2 = [1; z_2]; %Adding bias
        
        delta_2 = Theta2' * delta_3 .* sigmoidGradient(z_2);
    
        %Removing d2_0
        delta_2 = delta_2(2:end);
        
        Theta1_grad = Theta1_grad + delta_2 * a_1;
        Theta2_grad = Theta2_grad + delta_3 * a_2';
        
	end
	
    Theta1_grad = Theta1_grad./m;
    Theta2_grad = Theta2_grad./m;
    
	
	
	
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



    T1 = Theta1(:,2:end);
    T2 = Theta2(:,2:end);
    
    %Inserting zeros in the first column
    Theta1_with_theta_1_equal_to_0 = [zeros(size(T1,1), 1), T1];
    Theta2_with_theta_1_equal_to_0 = [zeros(size(T2,1), 1), T2];
    
    
    Theta1_grad = Theta1_grad + (lambda/m * Theta1_with_theta_1_equal_to_0);
    Theta2_grad = Theta2_grad + (lambda/m * Theta2_with_theta_1_equal_to_0);




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
