function Xn = normalize_angles(X,type)
% Put the angles in a certain range default is 0 : 180
if ~exist('type','var')
  type = 'deg2180';
end

switch type
  case 'deg2360'
    Xn = mod(X,360);
  case 'deg2180'
    Xn = mod(X,360);
    Xn(Xn>180) = 360 - Xn(Xn>180);
  case 'rad2deg'
    Xn = X/pi*180;
	case 'sin_cos'
		radX = deg2rad(X);
		Xn = [sin(radX) cos(radX)];
	case 'atan_sin_cos'
		Xn = rad2deg(atan2(X(:,1:size(X,2)/2),X(:,size(X,2)/2+1:end)));
	case 'none'
		Xn = X;
  otherwise
    error('Unknown!');
end
end