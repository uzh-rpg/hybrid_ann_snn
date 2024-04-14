function [D Dm] = H36MError(type, X, Y, posefeat, subject, distances)
% H36MError 
% This function computes the error. 
% X is the ground truth data and Y is the prediction.
% posefeat is the class for Y and subject is the subject from which Y
% comes from (necessary for computing mpjpd with joint angles for instance).

if size(X,1) ~= size(Y,1)
  error('Different number of examples!');
end

switch type
  case {'mpjpd','umpjpd'}
		if strcmp(type,'umpjpd')
			subject = H36MDataBase.instance().getUniversalSubject;
		end
		
		if strcmp(posefeat.Type, posefeat.ANGLES_TYPE)
			skel2 = subject.getAnglesSkel();
			if ~isempty(posefeat.Part)
				[~, skel2] = posefeat.select([], skel2, posefeat.Part);
			end
			
			for i = 1:size(Y,1)
				y = skel2xyz(skel2,Y(i,:))';

				Ytmp(i,:) = y(:);
			end
			Y = Ytmp;
		else
			skel = subject.getPosSkel();
			
			% keep the relevant joints
			if isempty(posefeat.Part)
				[Y ~] = posefeat.select(Y, skel, 'body');
			end
		end

		if size(X,2)~=size(Y,2)
			error('Different dimensionality!');
		end
		
		% mean per joint position distance
		Dm = zeros(size(X,1), size(X,2)/3);
		for i = 1: size(X,2)/3
			Dm(:,i) = sqrt(sum((X(:,(i-1)*3+1:i*3)-Y(:,(i-1)*3+1:i*3)).^2,2)); 
		end
		D = mean(Dm,2);
		
  case 'mpjad'
    % mean per joint angle distance
    skel = subject.getAnglesSkel();
		if isempty(posefeat.Part)
			[Y skel] = posefeat.select(Y, skel, 'body');
		end
		
		% ignoring translation
    Dm = abs(normalize_angles(X(:,4:end)-Y(:,4:end)));
		D = mean(Dm,2);
	
	case 'mpjad-nogr'
    % angle distance without global rotation
    skel = subject.getAnglesSkel();
		if isempty(posefeat.Part)
			[Y skel] = posefeat.select(Y, skel, 'body');
		end
		
		% ignoring translation and rotation
    Dm = abs(normalize_angles(X(:,7:end)-Y(:,7:end)));
		D = mean(Dm,2);
		
	case {'mpjle-mpjpd','mpjle-umpjpd'}
		if ~exist('distances','var')
			distances = 5:5:200;
		end
		
		if ~strcmp(type(7:end),'mpjpd') && ~strcmp(type(7:end),'umpjpd')
			error('MPJLE only works with umpjpd or mpjpd');
		end
		
		[~, Em] = H36MError(type(7:end), X, Y, posefeat, subject);
		
		for i = 1: length(distances)
			Dm(i,:,:) = Em<distances(i);	
			D(i,:) = mean(Em<distances(i),2);
		end
		
	case {'l2'}
		Dm = sqrt(sum((X-Y).^2,2));
		D = mean(Dm,2);
		
  otherwise
    error('What error is that ?!');
end