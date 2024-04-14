function NFeat = H36MNormalize(Feats,normtype,normdata,reverse)
	switch normtype
		case 'zscore'
			if ~exist('reverse','var')
				reverse=false;
			end
			
			if ~reverse
				NFeat = (Feats-ones(size(Feats,1),1)*normdata.mu)./(ones(size(Feats,1),1) * normdata.sigma);
				ind = normdata.sigma==0;
				NFeat(:,ind) = 0;
			else
				NFeat = Feats.*(ones(size(Feats,1),1) * normdata.sigma) +ones(size(Feats,1),1)*normdata.mu;
				ind = normdata.sigma==0;
				NFeat(:,ind) = 0;
			end
			
		case 'none'
			NFeat = Feats;
			
		otherwise
			error('Unknown type!');
	end
end