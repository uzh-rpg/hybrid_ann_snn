classdef H36MHoGFeature < H36MFeature
  % H36MHoGFeature compute a hog features
  % different types of masks and image colorspaces are supported
  
  properties
    MaskFeature;
    ImageFeature;
    
    Bins;
    Angle;
    PyramidLevels;
    
		scc;
		
    norm;
  end
  
  methods
    function obj = H36MHoGFeature(varargin)
      obj.Bins = 9;
      obj.Angle = 180;
      obj.PyramidLevels = 3;
      obj.norm = 'l1';
			obj.scc = true;
      obj = obj.fillin(varargin{:});
      
      obj.RequiredFeatures = {obj.MaskFeature, obj.ImageFeature};
      
      obj.FeatureName = ['back_mask_phog_nopb_' num2str(obj.Bins) '_orientations_' num2str(obj.PyramidLevels) '_levels'];

			if isempty(obj.MaskFeature)
				obj.MaskFeature = H36MMyBGMask;
			end
			obj.FeaturePath = ['/MyFeatures/' obj.MaskFeature.FeatureName '/'];

%       obj.Extension = '.mat';
			obj.Extension = '.cdf';
    end
    
    function [NFeat] = normalize(obj,Feat, ~, ~)
      switch obj.norm
        case 'l1'
          NFeat =  bsxfun(@rdivide, Feat,sum(Feat,2));
        otherwise
          error('Error !');
      end
    end
    
    function name = getMaskName(obj)
      name = obj.MaskFeature.getFeatureName();
    end
        
    function reader  = serializer(obj, Sequence)
			reader = H36MFeatureDataAccess([Sequence.getPath() filesep obj.FeaturePath Sequence.getName '__' obj.FeatureName obj.Extension],'Feat',false);
    end
    
    function [Feat] = process(obj,masks,images,~,~)
			% This is adapted from Joao Carreira's cpmc feature extraction code 
      if isa(obj.ImageFeature,'H36MRawVideoFeature')
        images = obj.ImageFeature.normalize(images);
			elseif isa(obj.ImageFeature,'H36MRGBVideoFeature')
				images = rgb2gray(images);
      elseif isa(obj.ImageFeature,'H36MGrayVideoFeature')
        % do nothing
      else
        error('Unknown image type!');
			end
      
			% get the relevant part of the mask (eliminate small components)
			[masks bbox] = preproc_mask(masks, obj.scc);
  
			% padding
      MARGIN = [10 10 10 10];

			bbox = [bbox(1:2) bbox(3:4)-1];
			
			props = regionprops(double(masks), 'BoundingBox');

      if(isempty(props))
        bbox(1:4) = [1 2 3 4];
      else
        % FIXME correct boundingbox
        for i = 1: length(props)
          boxes(i,:) = [props(i).BoundingBox(1:2) props(i).BoundingBox(1:2)+props(i).BoundingBox(3:4)];
        end
        mins = min(boxes,[],1); maxs = max(boxes,[],1);
        bb = [mins(1) mins(2) maxs(3)-mins(1) maxs(4)-mins(2)];

        bbox(1) = bb(2); %ymin
        bbox(2) = bbox(1) + bb(4); %ymax
        bbox(3) = bb(1); % xmin
        bbox(4) = bbox(3) + bb(3); % xmax
      end
      bbox = round(bbox);
      bbox(1) = max(bbox(1) - MARGIN(1), 1);
      bbox(2) = min(bbox(2) + MARGIN(2), size(masks,1));
      bbox(3) = max(bbox(3) - MARGIN(3), 1);
      bbox(4) = min(bbox(4) + MARGIN(4), size(masks,2));

      the_bboxes(:,1) = bbox';

      [Feat] = phog_backmasked(im2double(images), obj.Bins, obj.Angle, obj.PyramidLevels, the_bboxes, masks, edge(images, 'canny',[0.01 0.2]));

			Feat = single(Feat');
		end
	
		function size		= getFeatureSize(obj)
			size = obj.Bins*sum(4.^[0:obj.PyramidLevels]);
		end

	end
end