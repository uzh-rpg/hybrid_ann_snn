function D = H36MDist(type,X,Y)
% simple distance

switch type
	case 'l2'
		D = sqrt(sum((X-Y).^2,2));
	otherwise
		error('Unknown distance!');
end
end