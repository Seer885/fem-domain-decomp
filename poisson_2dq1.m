% MAIN FUNCTION
% num_nodes (Natural number, i.e. integer)

function poisson_2dq1(num_nodes)
    tic;

    % ====================================================================
    %% 0 - Numerical functions for later reference
    % ====================================================================
    % f-function in -\Delta u = f
    f = @(x,y) 2 .* (x + y) - 2 .* (x.^2 + y.^2);
    % Actual solution for error analysis
    u = @(x,y) x .* y .* (x - 1) .* (y - 1);

    % Q1 basis functions
    phi1 = @(x,y) (1-x) .* (1-y);
    phi2 = @(x,y) x .* (1-y);
    phi3 = @(x,y) x .* y;
    phi4 = @(x,y) (1-x) .* y;

    % Recurring floats saved for optimization
    a = 25/324;   b = 10/81;   c = (1-sqrt(3/5)) / 2;   d = (1+sqrt(3/5)) / 2;
    % Assemble quadrature weights and nodes
    Qweights = [a,b,a,b, 16/81 ,b,a,b,a];
    Qx_nodes = [c,c,c,1/2,1/2,1/2,d,d,d];
    Qy_nodes = [c,1/2,d,c,1/2,d,c,1/2,d];

    % Pre-compute outputs of Phi at quadrature nodes
    P = zeros(9,4);
    %GP = zeros(18,4);
    for q = 1:9
        xq = Qx_nodes(q);
        yq = Qy_nodes(q);
        P(q,:) = [phi1(xq,yq), phi2(xq,yq), phi3(xq,yq), phi4(xq,yq)];
    end


    % ====================================================================
    %% 1 - Q1 Setups
    % ====================================================================
    % Total number of rectangular elements for Q1
    num_elem = num_nodes^2;  
    num_nodes_all = num_nodes+1;  % Total nodes per variable in Q1

    xy_vals = linspace( 0 , 1 , num_nodes_all);  % List of all distinct values in Q1
    [X,Y] = meshgrid(xy_vals , xy_vals);  % Set up mesh grid for Q1

    % Assign a unique global index to each unique (i,j) in the above meshes
    % Will count up from top to bottom and then from left to right
    total_nodes = (num_nodes_all)^2;  % Total nodes period in Q1
    glob_idx = reshape( linspace(1, total_nodes, total_nodes), ...
        [num_nodes_all , num_nodes_all] );  % Q1


    % Create local connectivity arrays
    connections = zeros(4 , num_elem);

    % Q1: (1,2,3,4) --> (g1,g2,g3,g4)
    elem = 0;
    for i = 1:num_nodes
        for j = 1:num_nodes
            elem = elem + 1;

            % Starting at (xi,yj), 00 > 10 > 11 > 01
            connections(:,elem) = [glob_idx(i , j); glob_idx(i , j+1);
                glob_idx(i+1 , j+1); glob_idx(i+1 , j)];
        end
    end



    % ====================================================================
    %% 2 - Set up constant stiffness matrices and vectors
    % ====================================================================
    % Set up our consistent BK and its results
    h = 1 / num_nodes;   BK = [h , 0; 0 , h];   det_BK = h^2;

    % We have 1 global stiffness matrices and 1 global load vector:
    %          A; F
    % The local stiffness matrix for A is consistent thanks to our uniform distribution, so we can define it as so:
    lA = (1/6) * [4,-1,-2,-1 ; -1,4,-1,-2 ; -2,-1,4,-1 ; -1,-2,-1,4];

    % Meanwhile, we'll have to construct F the long way
    A = sparse(total_nodes,total_nodes); F = zeros(total_nodes,1);

    for elem = 1:num_elem
        % Set up load vectors for F
        lF = sparse(zeros(4,1));

        % Grab the corresponding 1st local node
        % This is needed for X = bK + BK X_hat
        [row,col] = find(glob_idx == connections(1,elem),1);
        bK = [X(row,col) ; Y(row,col)];

        % Set up F vector using Gaussian Quadratures
        for q = 1:9  % Quadrature index for weight/node
            Xg = bK + BK * [Qx_nodes(q) ; Qy_nodes(q)];

            lF(:) = lF(:) + Qweights(q) * f(Xg(1) , Xg(2)) * P(q,:)';
        end
        lF = det_BK * lF;


        % Obtain Q1 and Q2 global indices
        g_ind = connections(: , elem);

        % Update values of global stiffness matrices and load vectors
        A(g_ind,g_ind) = A(g_ind,g_ind) + lA;
        F(g_ind) = F(g_ind) + lF;
    end


    % Create 1D array of all QQ global indices on the boundary
    bound_idx = [glob_idx( 1 , : ) , ...
        glob_idx( num_nodes_all , : ) , ...
        glob_idx( 2:num_nodes_all-1 , 1 )' , ...
        glob_idx( 2:num_nodes_all-1 , num_nodes_all )'];
    % Set all rows and columns corresponding to bound indices to 0
    A(bound_idx , :) = 0;
    A(: , bound_idx) = 0;
    F(bound_idx) = 0;
    % Update corresponding diagonals of A to 1
    for idx = bound_idx(:)'
        A(idx , idx) = 1;
    end




    % ====================================================================
    %% 4 - Find u and then compute L2 error estimates by element
    % ====================================================================
    uh = A \ F;
    % This'll be the result of summing the L2 errors over the elements,
    % so we first set the total accumulated errors to 0
    u_l2 = 0;
    % Small optimization since going forward, we need det_BK * Qweights(q)
    Qweights = det_BK * Qweights;


    % Now we loop over each element to obtain individual L2 errors
    for elem = 1:num_elem
        % On the reference element:
        %   uh = uh(n1) * phi1 + uh(n2) * phi2 + ...
        UH = P * uh(connections(:,elem));

        % Grab the corresponding 1st local node (same for both Q2 and Q1)
        % This is needed for X = bK + BK X_hat
        [row,col] = find(glob_idx == connections(1,elem),1);
        bK = [X(row,col) ; Y(row,col)];

        for q = 1:9  % Quadrature index for weight/node
            Xg = bK + BK * [Qx_nodes(q) ; Qy_nodes(q)];  % Global nodes

            % Update L2 norms of ux, uy, and p
            u_l2 = u_l2 + Qweights(q) * (UH(q) - u(Xg(1),Xg(2)))^2;
        end
    end


    % Update our error since it's the square roots of the above sum
    u_l2 = sqrt(u_l2);

    % Print the results
    fprintf('u L2 Norm = %12.6f\n', u_l2);
    toc  % Output runtime
    fprintf('\n') 
end