% This program must be run by an external script initializing all variables

% PL = @(g0,d,a) 10^(g0*(d^-a)/10); % d in meters
% PL = @(g0,d,a) 10^(g0/10)*d^(-a); % d in meters

% [1] C. Hu, L. Dai, S. Han, and X. Wang, “Two-Timescale Channel Estimation
% for Reconfigurable Intelligent Surface Aided Wireless Communications,”
% IEEE Trans. Commun., vol. 69, no. 11, pp. 7736–7747, Nov. 2021,
% doi: 10.1109/TCOMM.2021.3072729.
% g0 = -20; % in dB
% d_AP_RIS = 20;
% a_AP_RIS = 2.1;
% d_AP_UE = 30;
% a_AP_UE = 2.2;
% d_UE_RIS = 20;
% a_UE_RIS = 4.2;

% [1] S. Lin, M. Wen, and F. Chen, “Cascaded Channel Estimation Using Full
% Duplex for IRS-Aided Multiuser Communications,” in IEEE Wireless
% Communications and Networking Conference, WCNC, 2022, vol. 2022-April,
% pp. 375–380, doi: 10.1109/WCNC51071.2022.9771718.

% Channels
% [1] K. Zheng, S. Ou, and X. Yin, “Massive MIMO channel models: A 
% survey,” International Journal of Antennas and Propagation, vol.  
% 2014. pp. 1–10, 2014, doi: 10.1155/2014/848071.

% [2] X. Chen, J. Shi, Z. Yang, and L. Wu, “Low-Complexity Channel 
% Estimation for Intelligent Reflecting Surface-Enhanced Massive MIMO,”
% IEEE Wirel. Commun. Lett., vol. 10, no. 5, pp. 996–1000, May 2021, 
% doi: 10.1109/LWC.2021.3054004.

if strcmp(channel_type, 'unstructured')
    % Rayleigh
    Hur = sqrt(1/2)*(randn(N,K) + 1j*randn(N,K)); % UE  -> RIS
    Hra = sqrt(1/2)*(randn(M,N) + 1j*randn(M,N)); % RIS -> AP
    Hua = sqrt(1/2)*(randn(M,K) + 1j*randn(M,K)); % UE  -> AP
elseif strcmp(channel_type, 'geometric')
    % geometric
    Hur = zeros(N,K);                                % UE  -> RIS
    for k = 1:K
        Hur(:,k) = generateLOSchannel([Nw,Nh],2, 1,1, LOS);
    end
    Hra = generateLOSchannel(M,1, [Nw,Nh],2, LOS);   % RIS -> AP
    Hua = zeros(M,K);                                % UE  -> AP
    for k = 1:K
        Hua(:,k) = generateLOSchannel(M,1, 1,1, LOS);
    end
end

% % Account for Scaling ambiguity
% if M > 1
%     if M > N
% %         if rank(Hra) == 1
%             Hra = Hra./Hra(:,1);
% %         else
% %             Hra(:,1) = ones(M,1);
% %         end
%     else
% %         if rank(Hra) == 1
%             Hra = Hra./Hra(1,:);
% %         else
% %             Hra(1,:) = ones(1,N);
% %         end
%     end
% end
% if K > 1
%     if K > N
% %         if rank(Hur) == 1
%             Hur = Hur./Hur(1,:);
% %         else
% %             Hur(1,:) = ones(1,K);
% %         end
%     else
% %         if rank(Hur) == 1
%             Hur = Hur./Hur(:,1);
% %         else
% %             Hur(:,1) = ones(N,1);
% %         end
%     end
% end

% Large Scale Fading: Path Loss
Hur = sqrt(g_ur)*Hur/norm(Hur,'fro');
Hra = sqrt(g_ra)*Hra/norm(Hra,'fro');
Hua = sqrt(g_ua)*Hua/norm(Hua,'fro');

% Channel Reciprocity
Hru = Hur.';
Har = Hra.';
Hau = Hua.';

% HurHra_kr = khatrirao(Hur.', Hra);

% % Vectorization of all channels
% h = [Hua(:); HurHra_kr(:)];
