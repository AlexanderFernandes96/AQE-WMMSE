% This script generates transmit - receive signal data for a wireless 
% RIS-assisted communication system along with optimal RIS phases and
% beamforming precoder
clear all; close all; delete(gcp('nocreate')); clc; 
TSTART = tic;
addpath("src")
%% Setup system model / script parameters
systemModelParameters

job_id = 0;
dataDir = "datasets/HDRISData/40PdBm/";
% rng(job_id)
dataDir = dataDir + num2str(job_id) + "/";
mkdir(dataDir);
fileSaveName = dataDir + "HDRISData";
dfile = fileSaveName + ".txt";
if exist(dfile, 'file') ; delete(dfile); end
diary(dfile)
diary on
vars2save =       {"channel_type", ...
                   "LOS", ...
                   ..."B", ...
                   ..."L", ...
                   ..."T", ...
                   "K", ...
                   "M", ...
                   "N", ...
                   "Nw", ...
                   "Nh", ...
                   ..."Xu", ...
                   ..."Psi", ...
                   ..."Hur_mc", ...
                   ..."Hra_mc", ...
                   ..."Hua_mc", ...
                   "Hru_mc", ...
                   "Har_mc", ...
                   "Hau_mc", ...
                   ..."g0", ...
                   "d_ra", ...
                   "a_ra", ...
                   "d_ur", ...
                   "a_ur", ...
                   "d_ua", ...
                   "a_ua", ...
                   "g_ur", ...
                   "g_ra", ...
                   "g_ua", ...
                   "CH_err", ...
                   "PdBm", ...
                   "NdBm", ...
                   "SNRdB", ...
                   ..."SINRdB", ...
                   "max_AO_iterations",...
                   "mc_runs"};

%% Generate pilots and RIS phase shifts
P = 10^(PdBm/10); % Transmission Power
SNR = 10^(SNRdB/10);

%% Monte Carlo
% channel matrices vectorized by stacking column vectors via H(:) operator
% then transposed as row vectors so that each row is a mc run
% Uplink:
% Hur_mc = complex(zeros(mc_runs,N*K));
% Hra_mc = complex(zeros(mc_runs,M*N));
% Hua_mc = complex(zeros(mc_runs,M*K));
% Downlink:
Hru_mc = complex(zeros(mc_runs,N*K));
Har_mc = complex(zeros(mc_runs,M*N));
Hau_mc = complex(zeros(mc_runs,M*K));
theta_mc = zeros(mc_runs,N); % optimal phase shifts
w_mc = zeros(mc_runs,M*K); % optimal beamforming
Ropt2_mc = zeros(mc_runs,1); % optimized receive signal (at the users)
Rrand2_mc = zeros(mc_runs,1); % random phases receive signal (at the users)

fprintf('Monte Carlo Run:\n');
mc_run_print = mc_runs;
for mc_run = 1:mc_runs
if mod(mc_run,mc_runs/mc_run_print) == 0
    fprintf('%i/%i\n', mc_run, mc_runs);
    fprintf("Script Execution time:\n")
    fprintElapsedTime(TSTART);
end
%% Create system model via channel matrices
generateHDRISchannels
% Creates Channel Matrices based on scructure of the channel
% Uplink:
%     Hur = (N,K) UE  -> RIS
%     Hra = (M,N) RIS -> AP
%     Hua = (M,K) UE  -> AP
% Downlink: channels reciprocity makes them matrix transpose equivalent
%     Hru = (K,N) RIS -> UE
%     Har = (N,M) AP  -> RIS
%     Hau = (K,M) AP  -> UE

%% Optimize phase shifts and beamforming 
% system model:
% SISO:
% Y = Hua*Xu + Hra*diag(exp(1j*theta))*Hur*Xu + N (Uplink)
% Y = Hau*Xa + Hru*diag(exp(1j*theta))*Har*Xa + N (Downlink)
% MISO:
% Y = W*(Hua + Hra*diag(exp(1j*theta))*Hur)*xu + N (Uplink)
% y = (hau + hru*diag(exp(1j*theta))*Har)*W*Xa + N (Downlink)

if M == 1 && K == 1 % SISO system model
    % No beamforming matrix only RIS phase shifts to optimize
    % Hua = 1x1 scalar
    % Hra = 1xN row vector
    % Hur = Nx1 column vector
    % SISO optimal phases are the same regardless of Uplink/Downlink
    % Solution is equation (21) from:
    % [1] Q. Wu, S. Zhang, B. Zheng, C. You, and R. Zhang, "Intelligent 
    % Reflecting Surface-Aided Wireless Communications: A Tutorial,” IEEE 
    % Trans. Commun., vol. 69, no. 5, pp. 3313–3351, May 2021, doi: 
    % 10.1109/TCOMM.2021.3051897.
    % modified using our 
    % (21) max_theta | hua + hra*diag(exp(1j*theta))*hur) |^2
    %      st. 0 <= theta <= 2*pi

    theta_opt = zeros(1,N);
    W_opt = 1;
    for n = 1:N
        % Knowledge of Perfect CSI
        a_kk = angle(Hua);
        b = angle(Hra(1,n));
        c = angle(Hur(n,1));
        theta_opt(1,n) = mod(a_kk - (b+c) + pi, 2*pi) - pi;
    end
    theta_rand = 2*pi*rand(1,N);
    Yopt = Hua + Hra*diag(exp(1j*theta_opt))*Hur;
    Yrand = Hua + Hra*diag(exp(1j*theta_rand))*Hur;

    Ropt = Yopt'*Yopt;
    Rrand = Yrand'*Yrand;

elseif K == 1 % Single-User MISO system model

    % find optimal phases for the single user
    hau = Hau(1,:);
    hru = Hru(1,:);
    Phi = diag(hru)*Har;
    R = [Phi*Phi', Phi*hau'; hau*Phi', 0];

    % to install cvx see: https://cvxr.com/cvx/doc/install.html
    cvx_begin quiet
        variable V(N+1,N+1) complex semidefinite
        maximize(trace(R*V))
        diag(V) == 1
    cvx_end

    % Gaussian random vector to obtain approximate best rank 1 vector
    [U,D] = eig(V);
    num_r = 1000; % number of SDR gaussian approximations
    r = 1/sqrt(2)*(rand(N+1,num_r) + 1j*rand(N+1,num_r));
    v = U*sqrt(D)*r;
    v = exp(1j*angle(v(1:N,1:num_r) ./ v(N+1,1:num_r)));
    v_init = v(:,1);
    Cost_init = 0;
    for n = 1:num_r
        Cost = 0;
        hau = Hau(1,:);
        hru = Hru(1,:);
        Cost = Cost + norm(hau + hru*diag(v(:,n))*Har);
        if Cost > Cost_init
            Cost_init = Cost;
            v_init = v(:,n);
            n_init = n;
        end
    end
    theta_opt = angle(v_init);

    y = hau + hru*diag(exp(1j*theta_opt(:,1)))*Har;
    W_opt(:,1) = y' / norm(y);

else % Multi-User MISO system model
    % Solve for beamforming matrix and RIS phase shifts
    % Hua = MxK scalar
    % Hra = MxN row vector
    % Hur = NxK column vector, K single antenna users
    % Solution can be found using Algorithm 1 in [6]
    
    %% Alternating Optimization Algorithm
    % [1] Q. Wu and R. Zhang, “Intelligent Reflecting Surface Enhanced 
    % Wireless Network via Joint Active and Passive Beamforming,” IEEE 
    % Trans. Wirel. Commun., vol. 18, no. 11, pp. 5394–5409, Nov. 2019, 
    % doi: 10.1109/TWC.2019.2936025.
    % [2] https://github.com/emilbjornson/optimal-beamforming/blob/master/functionFeasibilityProblem_cvx.m
    % [2] Z. Q. Luo and W. Yu, "An introduction to convex optimization 
    % for communications and signal processing,” IEEE J. Sel. Areas 
    % Commun., vol. 24, no. 8, pp. 1426–1438, Aug. 2006, 
    % doi: 10.1109/JSAC.2006.879347.
    % [3] A. Wiesel, Y. C. Eldar, and S. Shamai, "Linear precoding via 
    % conic optimization for fixed MIMO receivers,” IEEE Trans. Signal 
    % Process., vol. 54, no. 1, pp. 161–176, Jan. 2006, 
    % doi: 10.1109/TSP.2005.861073.
    % [4] Q. Shi, M. Razaviyayn, Z. Q. Luo, and C. He, "An iteratively 
    % weighted MMSE approach to distributed sum-utility maximization 
    % for a MIMO interfering broadcast channel," IEEE Trans. Signal 
    % Process., vol. 59, no. 9, pp. 4331–4340, Sep. 2011, 
    % doi: 10.1109/TSP.2011.2147784.
    % [5] H. Guo, Y. C. Liang, J. Chen, and E. G. Larsson, "Weighted 
    % Sum-Rate Maximization for Reconfigurable Intelligent Surface 
    % Aided Wireless Networks," IEEE Trans. Wirel. Commun., vol. 19, 
    % no. 5, pp. 3064–3076, May 2020, doi: 10.1109/TWC.2020.2970061.
    % [6] W. Jin, J. Zhang, C. K. Wen, S. Jin, X. Li, and S. Han, 
    % "Low-Complexity Joint Beamforming for RIS-Assisted MU-MISO 
    % Systems Based on Model-Driven Deep Learning," IEEE Trans. Wirel. 
    % Commun., vol. 23, no. 7, pp. 6968–6982, 2024, 
    % doi: 10.1109/TWC.2023.3336742.
    % [7] X. Yu, D. Xu, and R. Schober, "MISO wireless communication 
    % systems via intelligent reflecting surfaces: (Invited paper)," 2019 
    % IEEE/CIC Int. Conf. Commun. China, ICCC 2019, pp. 735–740, Aug. 2019,
    % doi: 10.1109/ICCCHINA.2019.8855810.

%     gammavar = 10^(SINRdB/10)*ones(K,1); % SINR constraint for all users
    R_best = 0;
    Ropt = zeros(K,1);
    Rrand = zeros(K,1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% 1. Initialize RIS phases and obtain first update of WMMSE variables
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % initialize RIS using [1] or randomization
    theta = 2*pi*rand(1,N); 
    
    % Complete communication channel given RIS phase shifts
    H = Hau + Hru*diag(exp(1j*theta))*Har;
    
    % Initialize W with zero-forcing
    W = H'*(H*H');
    W = sqrt(P)*W/norm(W,'fro');

    % Obtain initial update for WMMSE variables: U_, L_, and W
    U_ = zeros(K,1); % effective MMSE receiver (complex valued)
    L_ = zeros(K,1); % effective MMSE receiver weights (positive)

    % WMMSE algorithm update [6] (inspired by [4] and [5]) Update U_ and L_
    wwh = W(:,1)*W(:,1)';
    whw = W(:,1)'*W(:,1);
    for j = 2:K
        wwh = wwh + W(:,j)*W(:,j)';
        whw = whw + W(:,j)'*W(:,j);
    end
    for k = 1:K
        hk = H(k,:);
        U_(k) =  (hk*wwh*hk' + whw/SNR) \ hk*W(:,k);
        L_(k) = 1 / abs(1 - U_(k)'*hk*W(:,k));
    end

    % WMMSE algorithm update [6] (inspired by [4] and [5]) Update W
    for k = 1:K
        Sk = zeros(M,M);
        for j = 1:K
            Sk = Sk + abs(U_(j))^2 * L_(j) * (eye(M)/SNR + H(j,:)'*H(j,:));
        end
        W(:,k) = U_(k)*L_(k) * (Sk \ H(k,:)');
    end
    W = sqrt(P)*W/norm(W,'fro');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Begin iteratively solving WMMSE algorithm and RIS phase shifts
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for iter = 1:max_AO_iterations

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% 2. Update U_ and L_
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        wwh = W(:,1)*W(:,1)';
        whw = W(:,1)'*W(:,1);
        for j = 2:K
            wwh = wwh + W(:,j)*W(:,j)';
            whw = whw + W(:,j)'*W(:,j);
        end
        H = Hau + Hru*diag(exp(1j*theta))*Har;
        for k = 1:K
            hk = H(k,:);
            U_(k) =  (hk*wwh*hk' + whw/SNR) \ hk*W(:,k);
            L_(k) = 1 / abs(1 - U_(k)'*hk*W(:,k));
        end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% 3. Solve RIS phase shifts theta 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Solving problem P5 from [6] / Fixed point iteration from [7]
        A = zeros(N,N);
        b = zeros(N,1);
        wwh = W(:,1)*W(:,1)';
        for j = 2:K
            wwh = wwh + W(:,j)*W(:,j)';
        end
        for k = 1:K
            hau = Hau(k,:);
            hru = Hru(k,:);
            Hrk = diag(hru)*Har;
            A = A + L_(k)*abs(U_(k))^2*Hrk*wwh*Hrk';
            b = b - L_(k)*U_(k)'*Hrk*W(:,k) + L_(k)*abs(U_(k))^2*Hrk*wwh*hau';
        end
        Ab = [-A, -b; -(b'), 0];
%         R = Ab + norm(Ab,'fro')*eye(N+1); % [6]
        Abd = sort(eig(Ab), 'ascend', 'ComparisonMethod','real');
        R = Ab + (abs(Abd(1)) + 1e-6)*eye(N+1); % [7] make all eig(R) > 0
        
%         v = exp(1j*2*pi*[theta(:); 0]);
        % Relax optimal v to a norm constraint then applying unit modulus
        % to obtain initial value to the algorithm [7]
        [V,D] = eig(R);
        [d,ind] = sort(diag(D), 'descend');
        Vs = V(:,ind); % Sort eigenvectors
        v = sqrt(N+1)*Vs(:,1); % get largest eigenvector of R
        v = v ./ abs(v); % normalize with unit modulus constraint
        Rv = R*v;
        Rv_norm = norm(Rv, 1);
        n = 1;
        while(1)
            Rv_norm_prev = Rv_norm;
            v = Rv ./ abs(Rv); % [7]
            Rv = R*v;
            Rv_norm = norm(Rv, 1);
            v_ = v(1:N)'; % [7]
            n = n+1;
            
%             fprintf('%i, diff: %.8f\n', n, Rv_norm - Rv_norm_prev);
            if Rv_norm - Rv_norm_prev < 1e-6
%             fprintf('%i, diff: %.8f\n', n, Rv_norm - Rv_norm_prev);
                theta = angle(v_);
                break
            end
        end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% 4. Update Beamforming Matrix W 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        H = Hau + Hru*diag(exp(1j*theta))*Har;
        for k = 1:K
            Sk = zeros(M,M);
            for j = 1:K
                Sk = Sk + abs(U_(j))^2 * L_(j) * (eye(M)/SNR + H(j,:)'*H(j,:));
            end
            W(:,k) = U_(k)*L_(k) * (Sk \ H(k,:)');
        end
        W = sqrt(P)*W/norm(W,'fro');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% 5. Evaluate Sum Rate of iteration 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Rate = 0;
        for k = 1:K 
            hau = Hau(k,:);
            hru = Hru(k,:);
            hk = hau + hru*diag(exp(1j*theta))*Har;
            intf_opt = 10^(NdBm/10);
            for l = 1:K
                if l ~= k
                    intf_opt = intf_opt + norm(hk*W(:,l))^2;
                end
            end
            Rate = Rate + log2(1 + norm(hk*W(:,k))^2 / intf_opt);
            if Rate > R_best
                R_best = Rate;
                % Save optimal beamforming matrix and theta
                W_opt = sqrt(P)*W/norm(W,'fro');
                theta_opt = theta;
            end
        end
%         fprintf('iter %i  \tBest: %.4f \tRate: %.4f\n', iter, R_best, Rate);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% End Alternating Optimization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end % End AO
end

% Compare optimal phases vs random phases given optimal beamforming
% with true channels
for k = 1:K 
    hau = Hau(k,:);
    hru = Hru(k,:);
    theta_rand = 2*pi*rand(1,N) - pi;
    h_opt = hau + hru*diag(exp(1j*theta_opt))*Har;
    h_rand = hau + hru*diag(exp(1j*theta_rand))*Har;
    intf_opt = 10^(NdBm/10);
    intf_rand = 10^(NdBm/10);
    for l = 1:K
        if l ~= k
            intf_opt = intf_opt + norm(h_opt*W_opt(:,l))^2;
            intf_rand = intf_rand + norm(h_rand*W_opt(:,l))^2;
        end
    end
    Ropt(k) = log2(1 + norm(h_opt*W_opt(:,k))^2 / intf_opt);
    Rrand(k) = log2(1 + norm(h_rand*W_opt(:,k))^2 / intf_rand);
end

fprintf('R_opt: %.4f \tR_rand: %.4f\n', sum(Ropt), sum(Rrand));

theta_opt = theta_opt.'; % stack all phases into one row vector
w_opt = W_opt(:).';


% Example:     hur <=> Hur(:) for Hur = N by K matrix
% vectorize:   hur = reshape(Hur, [N*K,1]), stack columns into one column
% unvectorize: Hur = reshape(hur, [N,K]), unstack into K columns
Hru_mc(mc_run,:) = Hru(:).';
Har_mc(mc_run,:) = Har(:).';
Hau_mc(mc_run,:) = Hau(:).';
theta_mc(mc_run,:) = theta_opt.';  % optimized RIS phase shifts
w_mc(mc_run,:) = w_opt;  % optimized beamforming matrix
Ropt2_mc(mc_run,:) = sum(Ropt); % test receive signal is optimized
Rrand2_mc(mc_run,:) = sum(Rrand); % test receive signal is optimized
end % mc_run

%% Print
fprintf("\n------------------------------------------------------------\n")
fprintf("Mean Sum Rate over montecarlo runs:\n");
fprintf("Optimized RIS: %.4f +/- %.4f\n", mean(Ropt2_mc, 1), var(Ropt2_mc, 1));
fprintf("Random RIS: %.4f +/- %.4f\n", mean(Rrand2_mc, 1), var(Rrand2_mc, 1));
fprintf("Average Channel Error:\n")

%% Save data
save(fileSaveName + ".mat", vars2save{:})
systemModelParameters = [LOS, K, M, N, Nw, Nh, ...
    d_ra, a_ra, d_ur, a_ur, d_ua, a_ua, ...
    g_ur, g_ra, g_ua, CH_err, PdBm, NdBm, SNRdB, ...SINRdB,
    max_AO_iterations, mc_runs];
Tab = array2table(systemModelParameters);
Tab.Properties.VariableNames(1:length(systemModelParameters)) = ...
{'LOS', 'K', 'M', 'N', 'Nw', 'Nh', ...
 'd_ra', 'a_ra', 'd_ur', 'a_ur', 'd_ua', 'a_ua', ...
 'g_ur', 'g_ra', 'g_ua', 'CH_err', 'PdBm', 'NdBm', 'SNRdB', ...'SINRdB', 
 'max_AO_iterations', 'mc_runs'};
writetable(Tab, dataDir + "systemModelParameters.csv")

% Save channels into a single csv file
% % Uplink
% % real
% writematrix(real(Hur_mc), dataDir + "Hur_r.csv") 
% writematrix(real(Hra_mc), dataDir + "Hra_r.csv")
% writematrix(real(Hua_mc), dataDir + "Hua_r.csv")
% % imaginary
% writematrix(imag(Hur_mc), dataDir + "Hur_i.csv") 
% writematrix(imag(Hra_mc), dataDir + "Hra_i.csv")
% writematrix(imag(Hua_mc), dataDir + "Hua_i.csv")

% Downlink
% real
writematrix(real(Hru_mc), dataDir + "Hru_r.csv") 
writematrix(real(Har_mc), dataDir + "Har_r.csv")
writematrix(real(Hau_mc), dataDir + "Hau_r.csv")
% imaginary
writematrix(imag(Hru_mc), dataDir + "Hru_i.csv") 
writematrix(imag(Har_mc), dataDir + "Har_i.csv")
writematrix(imag(Hau_mc), dataDir + "Hau_i.csv")

% Save optimal RIS phase shifts (-pi <= theta < pi) and beamforming matrix
writematrix(theta_mc, dataDir + "RISopt.csv") % N RIS elements
writematrix(w_mc, dataDir + "beamforming.csv") % W is MxK, w = W(:).'
writematrix(real(w_mc), dataDir + "wopt_r.csv")
writematrix(imag(w_mc), dataDir + "wopt_i.csv")

delete(gcp('nocreate'))
fprintf("Script Execution time:\n")
fprintElapsedTime(TSTART);
diary off


