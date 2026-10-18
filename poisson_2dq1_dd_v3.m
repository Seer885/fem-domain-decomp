% MAIN FUNCTION
%
% num_subdoms (Natural number)
%  - Number of subdomains per variable (uniform distribution)
% num_loc_nodes (Natural number)
%  - Number of nodes per variable per subdomain *after* the first bottom-left node

function poisson_2dq1_dd_v3( num_subdoms , num_loc_nodes )
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
    loc_idx = reshape( 1:total_loc_nodes, ...
        num_loc_nodes1 , num_loc_nodes1 );
    glob_idx = reshape( 1:total_glob_nodes, ...
        num_glob_nodes1 , num_glob_nodes1 );


    % Create local and global connectivity arrays which also include the
    % location of the first node. Note that Q1: (1,2,3,4) --> (g1,g2,g3,g4)
    b1 = loc_idx(1:end-1 , 1:end-1);
    b2 = loc_idx(1:end-1 , 2:end);
    b3 = loc_idx(2:end , 2:end);
    b4 = loc_idx(2:end , 1:end-1);
    loc_connections = [b1(:)' ; b2(:)' ; b3(:)' ; b4(:)'];

    b1 = glob_idx(1:end-1 , 1:end-1);
    b2 = glob_idx(1:end-1 , 2:end);
    b3 = glob_idx(2:end , 2:end);
    b4 = glob_idx(2:end , 1:end-1);
    glob_connections = [b1(:)' ; b2(:)' ; b3(:)' ; b4(:)'];


    % Total interface (G) nodes
    tot_G = ((2 * num_loc_nodes - 1) * num_subdoms - 1) * (num_subdoms - 1);

    % For later steps, we need a specific ordering of interface nodes to
    % then map those to the global nodes. Given each row of G_to_glob is
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
    % Given each entry of the following array, the row is the global index
    % and the entry is:
    %   - 0 if it's not an interface node
    %   - The interface node index if the global index corresponds to one
    glob_to_G = zeros(total_glob_nodes,1);
    glob_to_G(G_to_glob) = 1:tot_G;



    % ====================================================================
    %% 2 - Set up B/G index info for each subdomain plus the F vector
    % ====================================================================
    % Set up our consistent BK and its results
    h = 1 / num_glob_nodes;   BK = [h , 0; 0 , h];   det_BK = h^2;
    % Multiplication constant array for later
    mult = det_BK * (Qweights .* P');  

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

    
    %loc_to_glob = zeros(1,total_loc_nodes);
    
    for j = 1 : num_loc_nodes : num_glob_nodes 
        for i = 1 : num_loc_nodes : num_glob_nodes
            dom_idx = dom_idx + 1;

            % Make a conversion array between local subdomain indices and 
            % global domain indices where the row is the corresponding 
            % local index and the entry is the global index
            loc_to_glob = glob_idx(i:i+num_loc_nodes , j:j+num_loc_nodes);
            Domains(dom_idx).loc_to_glob = loc_to_glob(:)';

            % The boundary (B) and interface (G) nodes on each subdomain 
            % vary in quantity and position depending on which subdomain
            % we're in. NINE possible situations can happen:
            %   - Corners (4 distinct setups)
            %   - Sides that aren't corners (4 distinct setups)
            %   - None of the above (1 distinct setup)
            % We'll want to save these local subdomain indices for each one
            n = num_glob_nodes-num_loc_nodes+1;
            %disp([j,i,num_glob_nodes-num_loc_nodes+1])
            % On the left side
            if j == 1
                % Bottom-left corner
                if i == 1
                    Domains(dom_idx).B_idx = [loc_idx( 1 , : ) , ...
                        loc_idx( 2:num_loc_nodes1 , 1 )'];
                    Domains(dom_idx).G_idx = setdiff(bound_idx, Domains(dom_idx).B_idx);
                % Top-left corner
                elseif i == n
                    Domains(dom_idx).B_idx = [loc_idx( num_loc_nodes1 , : ) , ...
                        loc_idx( 1:num_loc_nodes1-1 , 1 )'];
                    Domains(dom_idx).G_idx = setdiff(bound_idx, Domains(dom_idx).B_idx);
                % Not a corner, but still on left side
                else
                    Domains(dom_idx).B_idx = loc_idx( : , 1 )';
                    Domains(dom_idx).G_idx = setdiff(bound_idx, Domains(dom_idx).B_idx);
                end
            % On the right side
            elseif j == n
                % Bottom-right corner
                if i == 1
                    Domains(dom_idx).B_idx = [loc_idx( 1 , : ) , ...
                        loc_idx( 2:num_loc_nodes1 , num_loc_nodes1 )'];
                    Domains(dom_idx).G_idx = setdiff(bound_idx, Domains(dom_idx).B_idx);
                % Top-right corner
                elseif i == n
                    Domains(dom_idx).B_idx = [loc_idx( num_loc_nodes1 , : ) , ...
                        loc_idx( 1:num_loc_nodes1-1 , num_loc_nodes1 )'];
                    Domains(dom_idx).G_idx = setdiff(bound_idx, Domains(dom_idx).B_idx);
                % Not a corner, but still on right side
                else
                    Domains(dom_idx).B_idx = loc_idx( : , num_loc_nodes1 );
                    Domains(dom_idx).G_idx = setdiff(bound_idx, Domains(dom_idx).B_idx);
                end
            % On the bottom and not a corner
            elseif i == 1
                Domains(dom_idx).B_idx = loc_idx( 1 , : );
                Domains(dom_idx).G_idx = setdiff(bound_idx, Domains(dom_idx).B_idx);
            % On the top and not a corner
            elseif i == n
                Domains(dom_idx).B_idx = loc_idx( num_loc_nodes1 , : );
                Domains(dom_idx).G_idx = setdiff(bound_idx, Domains(dom_idx).B_idx);
            % Interior domain not touching the global boundary
            else
                Domains(dom_idx).B_idx = [];
                Domains(dom_idx).G_idx = bound_idx;
            end
            

            % Store the global x and y values on the current subdomain
            dom_X = X(i:i+num_loc_nodes , j:j+num_loc_nodes);
            dom_Y = Y(i:i+num_loc_nodes , j:j+num_loc_nodes);


            % Our F on each subdomain is directly dependent on the
            % subdomain, so we can't pre-emptively construct it.
            loc_F = sparse(total_loc_nodes,1);
            for elem = 1:num_loc_elem
                % Set up load vectors for F
                ref_F = sparse(4,1);

                %Grab the corresponding 1st local node
                %This is needed for X = bK + BK X_hat
                [row,col] = find(loc_idx == loc_connections(1,elem),1);
                bK = [dom_X(row,col) ; dom_Y(row,col)];

                % Set up F vector using Gaussian Quadratures
                for q = 1:9  % Quadrature index for weight/node
                    Xg = bK + BK * [Qx_nodes(q) ; Qy_nodes(q)];

                    ref_F(:) = ref_F(:) + mult(:,q) * f(Xg(1) , Xg(2));
                end


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
    I_idx = setdiff(1:total_loc_nodes , bound_idx);
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
        %disp(G_idx)
        cols = glob_to_G(Domains(d).loc_to_glob(G_idx));
        %disp(cols')
        R = sparse( 1:length(G_idx),cols , 1 , length(G_idx),tot_G );
        Domains(d).R = R;

        % (Will be useful later once we verify the current script works)
        Domains(d).A_IG = A_IG;
        Domains(d).A_GI = A_GI;
        Domains(d).A_GG = A_GG;
        Domains(d).F_I = F_I;
        Domains(d).F_G = F_G;
        Domains(d).C = C;
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
    % By construction, it will hold for each subdomain that:
    %       A_II u_I + A_IG u_G = f_I
    %       u_I = A_II^(-1) (f_I - A_IG u_G)
    % We can use this to avoid needing to apply boundary/interface
    % conditions to loc_A and loc_F on each subdomain!
    for d = 1:num_subdoms2
        loc_u_G = uh(Domains(d).loc_to_glob(Domains(d).G_idx));
        % Right-hand side before (theoretically) computing A_II^(-1)
        RHS = Domains(d).F_I - Domains(d).A_IG * loc_u_G;
        C = Domains(d).C;
        loc_u_I = C \ (C' \ RHS);

        % Update the values of our global solution matrix with the
        % corresponding values we just found on the subdomain
        uh(Domains(d).loc_to_glob(I_idx)) = loc_u_I;
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