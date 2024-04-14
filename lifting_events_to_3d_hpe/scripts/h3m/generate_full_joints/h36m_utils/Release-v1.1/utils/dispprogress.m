function dispprogress( text, curr, max, t )
% DISPPROGRESS(text,curr,max)
% text shown, curr [0..max] current index, max is the maximum index
	
	if curr < max
		if nargin == 3
			if curr > 0
				fprintf(repmat('\b',1,length(text)+5));
			end
			fprintf('%s %03d%%',text,ceil(curr/max*100));
		else
			if curr > 0
				fprintf(repmat('\b',1,length(text)+5+14));
			end
			fprintf('%s %03d%% ETA %8.02fs',text,ceil(curr/max*100),t/curr*(max-curr));
		end
	else
		if nargin == 3
			fprintf(repmat('\b',1,length(text)+5));
		else
			fprintf(repmat('\b',1,length(text)+5+14));
		end
	end
end

