function [a] = aURA(theta, psi, Mw, Mh)
%Uniform Rectangular Array steering vector, assume distance of spacing between 
% adjacent antennas is lambda/2.
%   theta - elevation angle in radians
%   psi - azimuth angle in radians
%   Mw - number of antennas (width)
%   Mh - numbe of antennas (height)

ax = zeros(Mw,1);
for m = 1:Mw
    ax(m) = exp(1j*pi*(m-1)*sin(theta)*sin(psi));
end
ay = zeros(Mh,1);
for m = 1:Mh
    ay(m) = exp(1j*pi*(m-1)*sin(theta)*cos(psi));
end

a = kron(ay,ax);

end