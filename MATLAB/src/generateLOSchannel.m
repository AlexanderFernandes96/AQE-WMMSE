function [H] = generateLOSchannel(M,aoa,N,aod,LOS)
%Generate M x N matrix with LOS paths
%   aoa and aod: 
%   1 for linear array, 2 for rectangular array
%   for rectuangular array use M = [Mw, Mh] or N = [Nw, Nh];
%   assume spacing between adjacent antennas is d = lambda/2


if aoa == 1
    Msize = M;
elseif aoa == 2
    Msize = M(1) * M(2);
else
    error("aoa must be 1, or 2.")
end
if aod == 1
    Nsize = N;
elseif aod == 2
    Nsize = N(1) * N(2);
else
    error("aoa must be 1, or 2.")
end
H = zeros(Msize,Nsize);

for l = 1:LOS
    h = sqrt(1/2)*(randn(1,1) + i*randn(1,1)); % path gain
    if aoa == 1
        theta = unifrnd(0,pi/2);
        AOA = aULA(theta,M);
    elseif aoa == 2
        theta = unifrnd(0,pi/2);
        psi = unifrnd(0,pi);
        AOA = aURA(theta, psi, M(1), M(2));
    end
    
    if aod == 1
        theta = unifrnd(0,pi/2);
        AOD = aULA(theta,N);
    elseif aod == 2
        theta = unifrnd(0,pi/2);
        psi = unifrnd(0,pi);
        AOD = aURA(theta, psi, N(1), N(2));
    end

    H = H + h*AOA*AOD';
end

end

