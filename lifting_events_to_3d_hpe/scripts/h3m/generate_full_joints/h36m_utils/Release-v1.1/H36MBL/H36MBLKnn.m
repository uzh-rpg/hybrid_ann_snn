classdef H36MBLKnn < H36MRegressor
  % knn
  % very simple nearest neighbors
	% types : vanilla, cover_tree
	% FIXME add ccv, 
	
  properties
    method; % mean, robust, iterative
    neighbors;
    dist;
    chunk_size;
    sample_rate;
		test_sample_rate;
    type;
		beta;
		radius;
    
    X;
    Y;
		
		MAX_SAMPLES;
  end
  
  methods
    function obj = H36MBLKnn(varargin)
					
			obj.type = 'vanilla';
			
			
			switch obj.type
				case 'vanilla'
					obj.method = 'mean';
					obj.neighbors = 1;
					obj.dist = {'chi2'};
					setenv('OMP_NUM_THREADS','8');
					obj.chunk_size = 2000;
					obj.beta = 10;
					obj.radius = .5;

					obj.test_sample_rate = 0;
					obj.MAX_SAMPLES = 300000;

					% modify the parameters
					obj = obj.fill_in(varargin{:});

					% subsample
					if obj.Ntex>obj.MAX_SAMPLES
						obj.sample_rate = floor(obj.Ntex/obj.MAX_SAMPLES);
					else
						obj.sample_rate = 1;
					end
			
					obj.X = [];
					obj.Y = [];
					
				otherwise
					error('Unknown type!');
			end
			
			if obj.neighbors == 1 && ~strcmp(obj.method,'mean')
				warning('Overriding to mean because only 1-nn!');
				obj.method = 'mean';
			end
			
			obj.traintime = 0;
			obj.testtime = 0;
    end
    
    function obj = train(obj)
			% NOOP 
    end
    
    function [Pred obj] = test(obj, X)			
			if obj.test_sample_rate ~= 0
				Ntot = size(X{1},1);
				X{1} = X{1}(obj.test_sample_rate/2:obj.test_sample_rate:end,:);
			end
			
      N = size(X{1},1);
			
			
			
			t = tic;
			switch obj.type
				case 'vanilla'
					ind = zeros(N*obj.neighbors,1);
					
					Pred = zeros(N,size(obj.Y,2));
					for i = 1: obj.chunk_size: N
						D = H36MDist2(obj.dist,X{1}(i:min(i+obj.chunk_size-1,N),:), obj.X);
						
						if obj.neighbors > 1
							% FIXME 
							[tmpD, tmpind] = sort(D,2);
							ind((i-1)*obj.neighbors+1:(i-1)*obj.neighbors+obj.neighbors*min(obj.chunk_size,N-i+1)) = reshape(tmpind(:,1:obj.neighbors)', [1 size(D,1)*obj.neighbors]);
						else
							[tmpD, tmpind] = min(D,[],2);
							ind = tmpind;
						end
						
						switch obj.method							
							case 'mean'
								if obj.neighbors > 1
									for j = 1: obj.neighbors
										%FIXME verify
										Pred(i:i+min(obj.chunk_size,N-i+1)-1,:) = Pred(i:i+min(obj.chunk_size,N-i+1)-1,:)+ obj.Y(tmpind(:,j),:); 
									end
									Pred(i:i+min(obj.chunk_size,N-i+1)-1,:) = Pred(i:i+min(obj.chunk_size,N-i+1)-1,:)/obj.neighbors;
								else
									Pred((i-1)*obj.neighbors+1:(i-1)*obj.neighbors+obj.neighbors*min(obj.chunk_size,N-i+1),:) = ...
										squeeze(reshape(obj.Y(ind,:), [obj.neighbors min(obj.chunk_size,N-i+1) size(obj.Y,2)])); 
								end
							otherwise
								error('Unknown method!');
						end
						
					end
					
				otherwise
					error('Unknown method!');
			end
			
			if obj.test_sample_rate ~= 0
				P = kron(Pred,ones(obj.test_sample_rate,1));
				if size(P,1)>Ntot
					Pred = P(1:Ntot,:);
				elseif Ntot-size(P,1)>0
					Pred = [P; repmat(P(end,:), Ntot-size(P,1),1)];
				else
					Pred = P;
				end
				if size(Pred,1)~=Ntot
					error('Prediction is bad!');
				end
			end
			
			obj.testtime = obj.testtime + toc(t);
    end
     
    function obj = update(obj, X, Y)
      if size(X,1) ~= size(Y,1)
				error('Inconsistent!');
			end
      
			t = tic;
			switch obj.type
				case 'vanilla'
					obj.X = [obj.X; X{1}(1:obj.sample_rate:end,:)];

					obj.Y = [obj.Y; Y{1}(1:obj.sample_rate:end,:)];
					
				otherwise
					error('Unknown type!');
			end
			
      obj.traintime = obj.traintime + toc(t);
    end
  end
end
