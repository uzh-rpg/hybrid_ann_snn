function [velocity acceleration nv na] = computeDynamics(posetype,P)

freq = 50;
% freq = 0.1269;

if isa(posetype,'H36MPose3DPositionsFeature') || isa(posetype,'H36MPoseUniversal3DPositionsFeature')
	ptm1 = P(1:end-2,:);
	pt	 = P(2:end-1,:);
	ptp1 = P(3:end,:);
	
	velocity = (ptp1-ptm1)/2/freq; % mm/s o(h^2)
	nv = sqrt(velocity.^2*kron(eye(size(P,2)/3),ones(3,1)));
% 	for i = 1: size(P,2)/3
% 		nv(:,i) = sqrt(sum(P(:,(i-1)*3+1:i*3).^2,2));
% 	end
	
	acceleration = (ptm1-2*pt + ptp1)/freq^2; % o(h^2)
	na = sqrt(acceleration.^2*kron(eye(size(P,2)/3),ones(3,1)));

% 		figure; hold on;
% 	quiver3(pt(:,1),pt(:,2),pt(:,3),velocity(:,1),velocity(:,2),velocity(:,3));
% 	quiver3(pt(:,1),pt(:,2),pt(:,3),acceleration(:,1),acceleration(:,2),acceleration(:,3),'r');
elseif isa(posetype,'H36MPose3DAnglesFeature')
	
elseif isa(posetype,'H36MPose2DPositionsFeature')
	ptm1 = P(1:end-2,:);
	pt	 = P(2:end-1,:);
	ptp1 = P(3:end,:);
	
	velocity = (ptm1-ptp1)/2/50; % mm/s o(h^2)
	acceleration = (ptm1-2*pt + ptp1)/50^2; % o(h^2)
end