% MAIN FUNCTION
%
% num_subdoms (Natural number)
%  - Number of subdomains per variable (uniform distribution)
% num_loc_nodes (Natural number)
%  - Number of nodes per variable per subdomain *after* the first bottom-left node

function poisson_2dq1_dd_v2( num_subdoms , num_loc_nodes )
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
    %% 1 - Setup for constructing our arrays
    % ====================================================================
    num_subdoms2 = num_subdoms^2;
    % Total nodes across all of our domain *after* (0,0)
    num_glob_nodes = num_loc_nodes * num_subdoms ;
    % Number of elements on each subdomain, then number of elements on the entire domain
    num_loc_elem = num_loc_nodes^2;
    num_glob_elem = num_loc_elem * num_subdoms2;  
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


    % Total interface (G) nodes
    tot_G = ((2 * num_loc_nodes - 1) * num_subdoms - 1) * (num_subdoms - 1);

    % For later steps, we need there to be a specific ordering of interface
    % nodes, and then map those to the global nodes. Given each row is
    % the index of an interface node, its entry is its global node.
    %        |     |
    %        3     6
    %        |     |
    %   --7--2--8--5--9--
    %        |     |
    %        1     4
    %        |     |
    G_to_glob = zeros(1,tot_G);
    % Add in column interfaces
    p = num_loc_nodes * num_subdoms - 1;
    for i = 1:(num_subdoms-1)
        G_to_glob(1 + p*(i-1):p*i) = ...
            glob_idx(2:num_loc_nodes * num_subdoms , num_loc_nodes*i+1)';
    end
    % Add in row interfaces (minus existing column ones)
    p = p - (num_subdoms - 1);
    q = nnz(G_to_glob);
    for i = 1:(num_subdoms-1)
        G_to_glob(q+1 + p*(i-1):q+p*i) = ...
            setdiff( ...
            glob_idx(num_loc_nodes*i+1 , 2:num_loc_nodes * num_subdoms),...
            G_to_glob);
    end



    % ====================================================================
    %% 2 - Set up B/G index info for each subdomain plus the F vector
    % ====================================================================
    % Set up our consistent BK and its results
    h = 1 / num_glob_nodes;   BK = [h , 0; 0 , h];   det_BK = h^2;

    % Save "boundary" indices on each subdomain
    bound_idx = [loc_idx( 1 , : ) , loc_idx( num_loc_nodes1 , : ) , ...
        loc_idx( 2:num_loc_nodes1-1 , 1 )' , ...
        loc_idx( 2:num_loc_nodes1-1 , num_loc_nodes1 )'];

    
    % Set up a structure for storing info on each subdomain, which saves:
    %   convert - Conversion between local and global indices
    %   B_idx - boundary indices
    %   G_idx - interface indices
    %   F - Stiffness array on each subdomain
    Domains = struct();
    dom_idx = 0;  % Shifting value for the current domain index
    % Make a conversion array between local subdomain indices and global
    % domain indices where the row is the corresponding local index
    loc_to_glob = zeros(1,total_loc_nodes);
    
    for j = 1 : num_loc_nodes : num_glob_nodes 
        for i = 1 : num_loc_nodes : num_glob_nodes
            dom_idx = dom_idx + 1;

            % Set the appropriate global indices in conversion array
            idx = 0;
            for jj = 1:num_loc_nodes1
                for ii = 1:num_loc_nodes1
                    idx = idx + 1;
                    loc_to_glob(idx) = glob_idx(i+ii-1,j+jj-1);
                end
            end
            Domains(dom_idx).loc_to_glob = loc_to_glob;

            % Indices of the "boundary" for all subdomains
            dom_bound_idx = [loc_idx( 1 , : ) , loc_idx( num_loc_nodes1 , : ) , ...
                loc_idx( 2:num_loc_nodes1-1 , 1 )' , ...
                loc_idx( 2:num_loc_nodes1-1 , num_loc_nodes1 )'];

            % The boundary (B) and interface (G) nodes on each subdomain 
            % vary in quantity and position depending on which subdomain
            % we're in. NINE possible situations can happen:
            %   - Corners (4 distinct setups)
            %   - Sides that aren't corners (4 distinct setups)
            %   - None of the above (1 distinct setup)
            % We'll want to save these local subdomain indices for each one

            % Bottom-left corner
            if dom_idx == 1  
                Domains(dom_idx).B_idx = [loc_idx( 1 , : ) , ...
                    loc_idx( 2:num_loc_nodes1 , 1 )'];
                Domains(dom_idx).G_idx = setdiff(dom_bound_idx, Domains(dom_idx).B_idx);
            % Top-left corner
            elseif dom_idx == num_subdoms  
                Domains(dom_idx).B_idx = [loc_idx( num_loc_nodes1 , : ) , ...
                    loc_idx( 1:num_loc_nodes1-1 , 1 )'];
                Domains(dom_idx).G_idx = setdiff(dom_bound_idx, Domains(dom_idx).B_idx);
            % Bottom-right corner
            elseif dom_idx == num_subdoms2 - num_subdoms + 1
                Domains(dom_idx).B_idx = [loc_idx( 1 , : ) , ...
                    loc_idx( 2:num_loc_nodes1 , num_loc_nodes1 )'];
                Domains(dom_idx).G_idx = setdiff(dom_bound_idx, Domains(dom_idx).B_idx);
            % Top-right corner
            elseif dom_idx == num_subdoms2
                Domains(dom_idx).B_idx = [loc_idx( num_loc_nodes1 , : ) , ...
                    loc_idx( 1:num_loc_nodes1-1 , num_loc_nodes1 )'];
                Domains(dom_idx).G_idx = setdiff(dom_bound_idx, Domains(dom_idx).B_idx);
            % Left side (excluding corners)
            elseif ismember(dom_idx,linspace(2,num_subdoms-1,num_subdoms-2))
                Domains(dom_idx).B_idx = loc_idx( : , 1 )';
                Domains(dom_idx).G_idx = setdiff(dom_bound_idx, Domains(dom_idx).B_idx);
            % Bottom side (excluding corners)
            elseif ismember(dom_idx,linspace(num_subdoms+1, ...
                    num_subdoms2-2*num_subdoms+1,num_subdoms-2))
                Domains(dom_idx).B_idx = loc_idx( 1 , : );
                Domains(dom_idx).G_idx = setdiff(dom_bound_idx, Domains(dom_idx).B_idx);
            % Top side (excluding corners)
            elseif ismember(dom_idx,linspace(2*num_subdoms, ...
                    num_subdoms2-num_subdoms,num_subdoms-2))
                Domains(dom_idx).B_idx = loc_idx( num_loc_nodes1 , : );
                Domains(dom_idx).G_idx = setdiff(dom_bound_idx, Domains(dom_idx).B_idx);
            % Right side (excluding corners)
            elseif ismember(dom_idx,linspace(num_subdoms2-num_subdoms+2,...
                    num_subdoms2-1,num_subdoms-2))
                Domains(dom_idx).B_idx = loc_idx( : , num_loc_nodes1 );
                Domains(dom_idx).G_idx = setdiff(dom_bound_idx, Domains(dom_idx).B_idx);
            % Interior domain not touching the global boundary
            else
                Domains(dom_idx).B_idx = [];
                Domains(dom_idx).G_idx = dom_bound_idx;
            end

            % Store the global x and y values on the current subdomain
            dom_X = X(i:i+num_loc_nodes , j:j+num_loc_nodes);
            dom_Y = Y(i:i+num_loc_nodes , j:j+num_loc_nodes);


            % Our F on each subdomain is directly dependent on the
            % subdomain, so we can't pre-emptively construct it.
            loc_F = sparse(total_loc_nodes,1);
            for elem = 1:num_loc_elem
                % Set up load vectors for F
                ref_F = sparse(zeros(4,1));

                %Grab the corresponding 1st local node
                %This is needed for X = bK + BK X_hat
                [row,col] = find(loc_idx == loc_connections(1,elem),1);
                bK = [dom_X(row,col) ; dom_Y(row,col)];

                % Set up F vector using Gaussian Quadratures
                for q = 1:9  % Quadrature index for weight/node
                    Xg = bK + BK * [Qx_nodes(q) ; Qy_nodes(q)];

                    ref_F(:) = ref_F(:) + Qweights(q) * f(Xg(1) , Xg(2)) * P(q,:)';
                end
                ref_F = det_BK * ref_F;


                % Obtain Q1 global indices
                l_ind = loc_connections(: , elem);

                % Update values of global stiffness matrices and load vectors
                loc_F(l_ind) = loc_F(l_ind) + ref_F;
            end

            Domains(dom_idx).F = loc_F;
        end 
    end



    % ====================================================================
    %% 3 - Construct A,S,g on each subdomain
    % ====================================================================
    % We have 1 global stiffness matrices and 1 global load vector:
    %          A; F
    % The reference element stiffness matrix for A is consistent thanks to 
    % our uniform distribution, so we can define it as so:
    ref_A = (1/6) * [4,-1,-2,-1 ; -1,4,-1,-2 ; -2,-1,4,-1 ; -1,-2,-1,4];

    % Our base A on every subdomain is identical as its values don't change
    % as a result of our boundary/interface conditions
    loc_A = sparse(total_loc_nodes,total_loc_nodes);
    for elem = 1:num_loc_elem
        l_ind = loc_connections(: , elem);

        % Update values of global stiffness matrices and load vectors
        loc_A(l_ind,l_ind) = loc_A(l_ind,l_ind) + ref_A;
    end

    % Furthermore, its interior values are always the same for all subdoms
    I_idx = setdiff([linspace(1, total_loc_nodes, total_loc_nodes)] , bound_idx);
    A_II = loc_A(I_idx, I_idx);
    C = chol(A_II);  % Save the cholesky factorization of A_II for later


    % However A_IG, A_GI, A_GG depend on the subdomain since G does not
    % include boundary indices on the full subdomain
    for d = 1:num_subdoms2
        % Depending on the current subdomain, grab A_II, 
        G_idx = Domains(d).G_idx;

        
        A_IG = loc_A(I_idx,G_idx);  
        A_GI = loc_A(G_idx,I_idx); 
        A_GG = loc_A(G_idx,G_idx);

        % The Schur Compliment (S) is:
        %       S = A_GG - A_GI (A_II)^{-1} A_IG
        % To avoid explicitly computing A_II's inverse, we instead solve:
        %       A_II Z = A_IG
        % using MatLab's solver to then set:
        %       S = A_GG - A_GI Z
        Z = C \ (C' \ A_IG);
        S = A_GG - A_GI * Z;
        Domains(d).S = S;


        F = Domains(d).F;
        F_I = F(I_idx);
        F_G = F(G_idx);

        % Meanwhile we need to construct:
        %       g = f_G - A_GI (A_II)^{-1} f_I
        % For similar reasons, we solve:
        %       A_II Z = f_I
        % such that we can instead set:
        %       g = f_G - A_GI Z
        Z = C \ (C' \ F_I);
        g = F_G - A_GI * Z;
        Domains(d).g = g;

        % Construct and save the R array for the current domain (height is
        % directly dependent on how many interface nodes there are)
        R = sparse(length(G_idx), tot_G);
        loc_to_glob = Domains(d).loc_to_glob;
        for i = 1:length(G_idx)
            idx = find(G_to_glob == loc_to_glob(G_idx(i)));
            R(i,  idx) = 1;
        end
        Domains(d).R = R;

        % (Will be useful later once we verify the current script works)
        Domains(d).A_IG = A_IG;
        Domains(d).A_GI = A_GI;
        Domains(d).A_GG = A_GG;
        Domains(d).F_I = F_I;
        Domains(d).F_G = F_G;
    end




    % ====================================================================
    %% 4 - Construct global S and g, then solve for u on the interface nodes
    % ====================================================================
    S = sparse(tot_G,tot_G);
    g = sparse(tot_G,1);
    for d = 1:num_subdoms2
        S = S + (Domains(d).R)' * Domains(d).S * Domains(d).R;
        g = g + (Domains(d).R)' * Domains(d).g;
    end
    

    % Thusly, we can find the values of u on our interface nodes:
    u_G = S \ g;
    uh = zeros(total_glob_nodes,1);
    uh(G_to_glob) = u_G;  % Update our solution values




    % ====================================================================
    %%  - Solve for u on the interior nodes (I)
    % ====================================================================
    % First we update our loc_A matrix since we now know the values on the
    % "boundary" (i.e. B and G) of any subdomain:
    loc_A(bound_idx , :) = 0;
    for idx = bound_idx(:)'
        loc_A(idx , idx) = 1;
    end


    % Iterate through each subdomain
    for d = 1:num_subdoms2
        loc_F = Domains(d).F;

        % Apply the boundary/interface values to loc_F as appropriate
        %   First we zero any boundary values in case they're actually
        %   on the boundary of the global domain
        loc_F(bound_idx) = 0;
        % Local subdomain indices for interface nodes
        inter_idx = Domains(d).G_idx;
        % Update 
        loc_F(inter_idx) = uh( Domains(d).loc_to_glob(inter_idx) );

        % With our subdomain arrays loc_A and loc_F contructed, solve 
        % the system loc_A loc_u = loc_F
        loc_u = loc_A \ loc_F;
        % Update the values of our global solution matrix with the
        % corresponding values we just found on the subdomain
        uh(Domains(d).loc_to_glob) = loc_u;
    end




    % ====================================================================
    %%  - Compute L2 error estimates by element
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
    t = toc;
    fprintf('Elapsed time is %f seconds.\n', t)
    %fprintf('Elapsed time with PC is around %f seconds.\n', t / num_subdoms2)
    fprintf('\n') 
end