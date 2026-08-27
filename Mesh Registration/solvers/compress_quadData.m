function q = compress_quadData( correspondences,weight )
% q = compress_quadData( correspondences )
% Given correspondences whose cost function is quadratic in the unknowns,
% compress the problem data into a single quadratic function.

% Initialize quadratic form
q = Quadratic(zeros(13));
% Accumulate quadratic form corresponding to each correspondence
for i=1:numel(correspondences)
%   correspondences(1,i).point.x
  q = q + weight(i)*quad( correspondences(i));
end

end
%#ok<*DQUAD>