% MAIN FUNCTION
%
% num_subdoms (Natural number)
%  - Number of subdomains per variable (uniform distribution)
% num_loc_nodes (Natural number)
%  - Number of nodes per variable per subdomain *after* the first bottom-left node

function poisson_2dq1_dd_v1( num_subdoms , num_loc_nodes )
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
    %% 1 - Setup for later constructing global A,F on entire domain and local A,f on each subdomain
    % ====================================================================
    % Total nodes across all of our domain *after* (0,0)
    num_glob_nodes = num_loc_nodes * num_subdoms ;
    % Number of elements on each subdomain, then number of elements on the entire domain
    num_loc_elem = num_loc_nodes^2;
    num_glob_elem = num_loc_elem * (num_subdoms^2);  
    % Total nodes per variable both locally and globally
    num_loc_nodes1 = num_loc_nodes+1;  
    num_glob_nodes1 = num_glob_nodes+1;  

    % List of all distinct x and y values in entire subdomain
    xy_vals = linspace( 0 , 1 , num_glob_nodes1);  
    [X,Y] = meshgrid(xy_vals , xy_vals);  % Set up mesh grid for Q1

    % Assign a unique global index to each unique (i,j) in the above meshes
    % Will count up from top to bottom and then from left to right
    total_loc_nodes = num_loc_nodes1^2;  % Total local nodes on each subdom
    total_glob_nodes = num_glob_nodes1^2;  % Total global nodes period 
    loc_idx = reshape( linspace(1, total_loc_nodes, total_loc_nodes), ...
        [num_loc_nodes1 , num_loc_nodes1] );
    glob_idx = reshape( linspace(1, total_glob_nodes, total_glob_nodes), ...
        [num_glob_nodes1 , num_glob_nodes1] );
    % Extract boundary indices on each subdomain
    bound_idx = [loc_idx( 1 , : ) , loc_idx( num_loc_nodes1 , : ) , ...
        loc_idx( 2:num_loc_nodes1-1 , 1 )' , ...
        loc_idx( 2:num_loc_nodes1-1 , num_loc_nodes1 )'];


    % Create local and global connectivity arrays
    loc_connections = zeros(4 , num_loc_elem);
    glob_connections = zeros(4 , num_glob_elem);

    % Q1: (1,2,3,4) --> (g1,g2,g3,g4)
    elem = 0;
    for i = 1:num_loc_nodes
        for j = 1:num_loc_nodes
            elem = elem + 1;

            % Starting at (xi,yj), 00 > 10 > 11 > 01
            loc_connections(:,elem) = [loc_idx(i , j); loc_idx(i , j+1);
                loc_idx(i+1 , j+1); loc_idx(i+1 , j)];
        end
    end
    elem = 0;
    for i = 1:num_glob_nodes
        for j = 1:num_glob_nodes
            elem = elem + 1;

            % Starting at (xi,yj), 00 > 10 > 11 > 01
            glob_connections(:,elem) = [glob_idx(i , j); glob_idx(i , j+1);
                glob_idx(i+1 , j+1); glob_idx(i+1 , j)];
        end
    end

    
    % Extract global indices of:
    %       interior nodes (I), interface nodes (G), and boundary nodes (B)
    if num_loc_nodes == 2
        start = 1 : 2 : num_glob_nodes;
        I_idx = reshape([ glob_idx(start+1,start+1)], 1, []);
    elseif num_loc_nodes == 3
        start = 1 : 3 : num_glob_nodes;
        I_idx = reshape([ glob_idx(start+1,start+1), ...
            glob_idx(start+2,start+1), ...
            glob_idx(start+1,start+2), ...
            glob_idx(start+2,start+2)], 1, []);
    else
        I_idx = [];

        for j = 1 : num_loc_nodes : num_glob_nodes
            for i = 1 : num_loc_nodes : num_glob_nodes

                loc_I = glob_idx(i+1 : i+num_loc_nodes-1, ...
                    j+1 : j+num_loc_nodes-1);

                I_idx = [I_idx, reshape(loc_I(:),1,[])];
            end
        end
    end

    B_idx = [glob_idx( 1 , : ) , glob_idx( num_glob_nodes1 , : ) , ...
        glob_idx( 2:num_glob_nodes1-1 , 1 )' , ...
        glob_idx( 2:num_glob_nodes1-1 , num_glob_nodes1 )'];

    G_idx = setdiff([linspace(1, total_glob_nodes, total_glob_nodes)] ,...
        [I_idx B_idx]);
    I_idx = sort(I_idx);




    % ====================================================================
    %% 2 - Construct global A and F, then extract our critical blocks
    % ====================================================================
    % Set up our consistent BK and its results
    h = 1 / num_glob_nodes;   BK = [h , 0; 0 , h];   det_BK = h^2;

    % We have 1 global stiffness matrices and 1 global load vector:
    %          A; F
    % The local stiffness matrix for A is consistent thanks to our uniform distribution, so we can define it as so:
    lA = (1/6) * [4,-1,-2,-1 ; -1,4,-1,-2 ; -2,-1,4,-1 ; -1,-2,-1,4];

    % Meanwhile, we'll have to construct F the long way
    A = sparse(total_glob_nodes,total_glob_nodes); 
    F = zeros(total_glob_nodes,1);

    for elem = 1:num_glob_elem
        % Set up load vectors for F
        lF = sparse(zeros(4,1));

        % Grab the corresponding 1st local node
        % This is needed for X = bK + BK X_hat
        [row,col] = find(glob_idx == glob_connections(1,elem),1);
        bK = [X(row,col) ; Y(row,col)];

        % Set up F vector using Gaussian Quadratures
        for q = 1:9  % Quadrature index for weight/node
            Xg = bK + BK * [Qx_nodes(q) ; Qy_nodes(q)];

            lF(:) = lF(:) + Qweights(q) * f(Xg(1) , Xg(2)) * P(q,:)';
        end
        lF = det_BK * lF;


        % Obtain Q1 global indices
        g_ind = glob_connections(: , elem);

        % Update values of global stiffness matrices and load vectors
        A(g_ind,g_ind) = A(g_ind,g_ind) + lA;
        F(g_ind) = F(g_ind) + lF;
    end

    A_II = A(I_idx,I_idx);   A_IG = A(I_idx,G_idx);
    A_GI = A(G_idx,I_idx);   A_GG = A(G_idx,G_idx);
    f_I = F(I_idx);   f_G = F(G_idx);




    % ====================================================================
    %% 3 - Solve for u on the interface nodes (G)
    % ====================================================================
    % The Schur Compliment (S) is:
    %       S = A_GG - A_GI (A_II)^{-1} A_IG
    % To avoid explicitly computing A_II's inverse, we instead solve:
    %       A_II Z = A_IG
    % using MatLab's solver to then set:
    %       S = A_GG - A_GI Z
    R = chol(A_II);
    Z = R \ (R' \ A_IG);
    S = A_GG - A_GI * Z;

    % Meanwhile we need to construct:
    %       g = f_G - A_GI (A_II)^{-1} f_I
    % For similar reasons, we solve:
    %       A_II Z = f_I
    % such that we can instead set:
    %       g = f_G - A_GI Z
    Z = R \ (R' \ f_I);
    g = f_G - A_GI * Z;

    % Thusly, we can find the values of u on our interface nodes:
    u_G = S \ g;
    uh = zeros(total_glob_nodes,1);
    uh(G_idx) = u_G;  % Update our solution values




    % ====================================================================
    %% 4 - Solve for u on the interior nodes (I)
    % ====================================================================
    % Note that this is optimally done with parallel computing, but we
    % currently do not have access to that, so computing the interior nodes
    % on each subdomain will (for now) need to be done on a for loop

    % Make a conversion array between local subdomain indices and global
    % domain indices where the column is the corresponding local index
    convert = zeros(1,total_loc_nodes);

    % Our base A on every subdomain is identical as its values don't change
    % as a result of our boundary/interface conditions
    dA = sparse(total_loc_nodes,total_loc_nodes);
    for elem = 1:num_loc_elem
        l_ind = loc_connections(: , elem);

        % Update values of global stiffness matrices and load vectors
        dA(l_ind,l_ind) = dA(l_ind,l_ind) + lA;
    end
    dA(bound_idx , :) = 0;
    %dA(: , bound_idx) = 0;
    for idx = bound_idx(:)'
        dA(idx , idx) = 1;
    end


    % Iterate through the bottom-left node of each subdomain
    for j = 1 : num_loc_nodes : num_glob_nodes 
        for i = 1 : num_loc_nodes : num_glob_nodes
            
            % Set the appropriate global indices in conversion array
            idx = 0;
            for jj = 1:num_loc_nodes1
                for ii = 1:num_loc_nodes1
                    idx = idx + 1;
                    convert(idx) = glob_idx(i+ii-1,j+jj-1);
                end
            end

            % Fetch the global x and y values on this subdomain
            dX = X(i:i+num_loc_nodes , j:j+num_loc_nodes);
            dY = Y(i:i+num_loc_nodes , j:j+num_loc_nodes);
            
            % Our F on each subdomain is directly dependent though on the
            % subdomain, so we can't pre-emptively construct it.
            dF = sparse(total_loc_nodes,1);
            for elem = 1:num_loc_elem
                % Set up load vectors for F
                lF = sparse(zeros(4,1));

                %Grab the corresponding 1st local node
                %This is needed for X = bK + BK X_hat
                [row,col] = find(loc_idx == loc_connections(1,elem),1);
                bK = [dX(row,col) ; dY(row,col)];

                % Set up F vector using Gaussian Quadratures
                for q = 1:9  % Quadrature index for weight/node
                    Xg = bK + BK * [Qx_nodes(q) ; Qy_nodes(q)];

                    lF(:) = lF(:) + Qweights(q) * f(Xg(1) , Xg(2)) * P(q,:)';
                end
                lF = det_BK * lF;


                % Obtain Q1 global indices
                l_ind = loc_connections(: , elem);

                % Update values of global stiffness matrices and load vectors
                dF(l_ind) = dF(l_ind) + lF;
            end

            % Apply the boundary/interface values to F as appropriate
            %   First we zero any boundary values in case they're actually
            %   on the boundary of the global domain
            dF(bound_idx) = 0;
            % Local subdomain indices for interface nodes
            inter_idx = find(ismember(convert , G_idx));
            dF(inter_idx) = uh(convert(inter_idx));

            % With our subdomain arrays dA and dF contructed, solve the
            % system dA du = dF
            %R = chol(dA);
            %du = R \ (R' \ dF);
            du = dA \ dF;
            % Update the values of our global solution matrix with the
            % corresponding values we just found on the subdomain
            uh(convert) = du;
        end 
    end




    % ====================================================================
    %% 5 - Compute L2 error estimates by element
    % ====================================================================
    % This'll be the result of summing the L2 errors over the elements,
    % so we first set the total accumulated errors to 0
    u_l2 = 0;
    % Small optimization since going forward, we need det_BK * Qweights(q)
    Qweights = det_BK * Qweights;


    % Now we loop over each element to obtain individual L2 errors
    for elem = 1:num_glob_elem
        % On the reference element:
        %   uh = uh(n1) * phi1 + uh(n2) * phi2 + ...
        UH = P * uh(glob_connections(:,elem));

        % Grab the corresponding 1st local node
        % This is needed for X = bK + BK X_hat
        [row,col] = find(glob_idx == glob_connections(1,elem),1);
        bK = [X(row,col) ; Y(row,col)];

        for q = 1:9  % Quadrature index for weight/node
            Xg = bK + BK * [Qx_nodes(q) ; Qy_nodes(q)];  % Global nodes

            % Update L2 norms of ux, uy, and p
            u_l2 = u_l2 + Qweights(q) * (UH(q) - u(Xg(1),Xg(2)))^2;
        end
    end


    % Update our error since it's the square root of the above sum
    u_l2 = sqrt(u_l2);

    % Print the results
    fprintf('u L2 Norm = %12.6f\n', u_l2);
    toc  % Output runtime
    fprintf('\n') 
end