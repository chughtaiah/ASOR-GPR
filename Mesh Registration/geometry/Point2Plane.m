classdef Point2Plane < Point2X
  %Point2Plane Correspondence between 3D point and 3D plane
  %   See also Correspondence.
  
  %#ok<*PROPLC>
  
  properties
    % inherited from Point2X
    format = 'bx';
  end
  
  methods
    function this = Point2Plane( point,modelPlane )
      % correspondence = Point2Plane( POINT,MODEL_PLANE )
      %   Constructor from components
      
      if nargin == 0
        % default values
        point = Point();
        modelPlane = Plane();
      end

      this.point = point;
      this.model = modelPlane;
    end
    
    function q = quad(this)
      % Return quadratic form for this correspondence (in terms of vec(T))
%       disp('Xy');
      % Define point concatenation matrix
      Xy = [ kron(makehom(this.point.x)',eye(3)), -this.model.x ];
      % Define central matrix for point-to-model
      this.model.n;
%       disp('C');
      C  = this.model.n*this.model.n';
%       
%       disp('Q');
      % Create homogeneized quadratic in [vec(R),t,1]=[vec(T),1]
      Q_ = symmetrize(Xy'*C*Xy); % symmetrize for num stability
      q = Quadratic(Q_);
    end
    
  end
end
