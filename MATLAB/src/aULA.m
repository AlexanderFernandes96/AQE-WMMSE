function [a] = aULA(theta, M)
%Uniform Linear Array steering vector, assume distance of spacing between 
% adjacent antennas is lambda/2.
%   theta - angle in radians
%   M - number of antennas

a = zeros(M,1);
for m = 1:M
    a(m) = exp(1j*pi*(m-1)*sin(theta));
end

end

