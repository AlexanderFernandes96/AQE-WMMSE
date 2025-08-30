function [TEND] = fprintElapsedTime(TSTART)
%Print Elapsed Time
TEND = toc(TSTART);
fprintf('Elapsed time: %d days, %d hours, %d minutes, %f seconds\n', ...
    floor(TEND/(60*60*24)), floor(rem(TEND,60*60*24)/(60*60)), floor(rem(TEND,60*60)/60), rem(TEND,60));
end