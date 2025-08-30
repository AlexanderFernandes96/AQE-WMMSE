function [C] = khatrirao(A,B)
%Khatri-Rao Product
% A is I x K
% B is J x K
% C = khatrirao(A,B) is IJ x K

K = size(A,2);
J = size(B,1);
C = reshape(B,[J 1 K]);
I = size(A,1);
A = reshape(A,[1 I K]);
C = reshape(bsxfun(@times,A,C),[I*J 1 K]);
C = reshape(C,[size(C,1) K]);

end

% function X = khatrirao(U,varargin)
% %KR Khatri-Rao product.
% %   kr(A,B) returns the Khatri-Rao product of two matrices A and B, of 
% %   dimensions I-by-K and J-by-K respectively. The result is an I*J-by-K
% %   matrix formed by the matching columnwise Kronecker products, i.e.
% %   the k-th column of the Khatri-Rao product is defined as
% %   kron(A(:,k),B(:,k)).
% %
% %   kr(A,B,C,...) and kr({A B C ...}) compute a string of Khatri-Rao 
% %   products A o B o C o ..., where o denotes the Khatri-Rao product.
% %
% %   See also kron.
% %   Version: 21/10/10
% %   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
% if ~iscell(U), U = [U varargin]; end
% K = size(U{1},2);
% if any(cellfun('size',U,2)-K)
%     error('kr:ColumnMismatch', ...
%           'Input matrices must have the same number of columns.');
% end
% J = size(U{end},1);
% X = reshape(U{end},[J 1 K]);
% for n = length(U)-1:-1:1
%     I = size(U{n},1);
%     A = reshape(U{n},[1 I K]);
%     X = reshape(bsxfun(@times,A,X),[I*J 1 K]);
%     J = I*J;
% end
% X = reshape(X,[size(X,1) K]);
% end


% function [C] = khatrirao(A,B)
% %Khatri-Rao Product
% % A is I x K
% % B is J x K
% % C = khatrirao(A,B) is IJ x K
% 
% [I,J] = size(A);
% [J,~] = size(B);
% C = zeros(J*I,J);
% for n = 1:J
%     C(:,n) = kron(A(:,n), B(:,n));    
% end
% end