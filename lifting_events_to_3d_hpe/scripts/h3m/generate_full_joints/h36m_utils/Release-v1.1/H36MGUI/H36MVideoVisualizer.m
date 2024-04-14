classdef H36MVideoVisualizer
  properties
    ShowMask;
    ShowPose;
    ShowResult;
    ShowImage;
    
		MirrorSymmetric;
		
		Frequency;
		
    Masks;
    VideoFeatures;
    Skeletons;
    PoseFeatures;
    
    VideoData;
    InputFileNames;
    MaskData;
		BBMaskData;
		MaskFlip;
    PoseData;
    PoseFileNames;
    
    NumFrames;
    
    Subject;
    Action;
    SubAction;
    Cameras;
    Skel;
    
    Resize;
    
    CurrentFrame;
    
    PoseHandles;
    ScenePoseHandles;
    
    DepthData;
    Sequences;
    
    Views;
    NumViews;
    
    SceneHandles;
    DepthHandles;
    
    XLimits;
    YLimits;
    ZLimits;
    
    Cams;
    Length;
    
    Predictions;
		InputDataAccess;
  end
  
  methods
    function obj = H36MVideoVisualizer()
      obj.ShowMask = [1 1 1 1];
      obj.ShowPose = [1 1 1 1];
      obj.ShowResult = [1 1 1 1];
			obj.MirrorSymmetric = [0 0 0 0];
      obj.ShowImage = true;
      obj.Resize = .4;
      
      obj.Masks{1} = H36MFullMask();
      obj.Masks{2} = H36MMyBGMask('Resize',obj.Resize,'scc',1);
      obj.Masks{3} = H36MMyBBMask('Resize',obj.Resize);
      obj.Masks{4} = H36MMyBGMask('Resize',obj.Resize,'Symmetric',1,'scc',1);
      obj.Masks{5} = H36MMyBBMask('Resize',obj.Resize,'Symmetric',1);
			
      obj.VideoFeatures{1} = H36MRGBVideoFeature('Resize',obj.Resize);
			obj.VideoFeatures{2} = H36MRGBVideoFeature('Resize',obj.Resize,'Symmetric',1);
      
      obj.PoseFeatures{1} = H36MNoPose();
			obj.PoseFeatures{2} = H36MPose3DAnglesFeature();
      obj.PoseFeatures{3} = H36MPose3DPositionsFeature();
      obj.PoseFeatures{4} = H36MPose2DPositionsFeature('Dimensions',2);
			obj.PoseFeatures{5} = H36MPose3DAnglesFeature('Symmetric',1);
      obj.PoseFeatures{6} = H36MPose3DPositionsFeature('Symmetric',1);
      obj.PoseFeatures{7} = H36MPose2DPositionsFeature('Dimensions',2,'Symmetric',1);

      obj.Skel{1} = [];
      obj.Skel{2} = [];
      obj.Skel{3} = [];
            
      obj.NumViews = 0;
      obj.Frequency = 50;
			
      obj.Views = {};
      obj.PoseHandles= cell(1,4);
            
      obj.ScenePoseHandles = [];
    end
    
    function obj = load(obj, s, a, sa)
      db = H36MDataBase.instance();
      global APPLICATION;
            
      obj.CurrentFrame = 1;
      obj.Subject = db.getSubject(s);
      obj.Action = a;
      obj.SubAction = sa;
          
      obj.Skeletons{1} = obj.Subject.getAnglesSkel;
      obj.Skeletons{2} = obj.Subject.getPosSkel;
			
      obj.Skel{2} = obj.Subject.get2DPosSkel();
      obj.Skel{3} = obj.Subject.get2DPosSkel();
      obj.Skel{4} = obj.Subject.get2DPosSkel();
			obj.Skel{5} = obj.Subject.get2DPosSkel();
      obj.Skel{6} = obj.Subject.get2DPosSkel();
      obj.Skel{7} = obj.Subject.get2DPosSkel();
			
      obj.NumFrames = db.getNumFrames(s,a,sa);
      obj.Length = 1000;
      
      obj.MaskData = cell(4,1);
			obj.BBMaskData = cell(4,1);
			obj.PoseData = cell(7,4);
			obj.ShowPose = [1 1 1 1];
			obj.MirrorSymmetric = [0 0 0 0];
			obj.MaskFlip = logical([0 0 0 0]);
			if isempty(obj.InputDataAccess)
				obj.InputDataAccess = cell(1,4);
			end
			
			obj.DepthData = {};
			
      for c = 1: 4
        obj.Cameras{c} = db.getCamera(s,c);
        obj.Sequences{c} = H36MSequence(s, a, sa, c);
				obj.Views{c}.mask = 1;
				if ~isempty(obj.PoseHandles{c})
					delete(obj.PoseHandles{c}(2:end));
					obj.PoseHandles{c} = [];
				end
				if ~isempty(obj.InputDataAccess{c})
					obj.InputDataAccess{c}.delete;
					obj.InputDataAccess{c} = [];
				end
			end
			
      set(APPLICATION.handles.noframes,'string',num2str(obj.NumFrames));
      
      obj.XLimits = [-3000 5000];
      obj.YLimits = [-6000 6000];      
      obj.ZLimits = [    0 3000];
    end
    
    function showCameras(obj)
%       figure(obj.SceneHandles.figure1); %hold on; axis equal; grid on;
      zlim(obj.ZLimits), xlim(obj.XLimits), ylim(obj.YLimits), xlabel('X'), ylabel('Y'), zlabel('Z'); grid on;
      axisorder = [1 2 3];rotate3d on;
      hold on;
      for i = 1: length(obj.Cameras)
        O = obj.Cameras{i}.T;
        
        VX(1,:) = [obj.Length/3 0 0] * obj.Cameras{i}.R;
        VX(2,:) = [0 obj.Length/3 0] * obj.Cameras{i}.R;
        VX(3,:) = [0 0 obj.Length] * obj.Cameras{i}.R;
        
        VO(1,:) = O + VX(1,:);
        VO(2,:) = O + VX(2,:);
        VO(3,:) = O + VX(3,:);
        
        VN(1,:) = VX(1,:) + VX(2,:) + VX(3,:) + O;
        VN(2,:) = VX(1,:) - VX(2,:) + VX(3,:) + O;
        VN(3,:) = -VX(1,:)+ VX(2,:) + VX(3,:) + O;
        VN(4,:) = -VX(1,:)- VX(2,:) + VX(3,:) + O;
        text(O(axisorder(1)),O(axisorder(2)),O(axisorder(3)),['Camera' num2str(obj.Cameras{i}.Number)]);
%         plot3([O(axisorder(1)) VO(1,axisorder(1))],[O(axisorder(2)) VO(1,axisorder(2))],[O(axisorder(3)) VO(1,axisorder(3))],'LineWidth',2,'Color','r');
%         plot3([O(axisorder(1)) VO(2,axisorder(1))],[O(axisorder(2)) VO(2,axisorder(2))],[O(axisorder(3)) VO(2,axisorder(3))],'LineWidth',2,'Color','g');
        plot3([O(axisorder(1)) VO(3,axisorder(1))],[O(axisorder(2)) VO(3,axisorder(2))],[O(axisorder(3)) VO(3,axisorder(3))],'LineWidth',2,'Color','r');
        line(VN([1 2 4 3 1],1),VN([1 2 4 3 1],2),VN([1 2 4 3 1],3),'LineWidth',2,'Color',.7*[1 1 1]);
        line([O(1) VN(1,1)],[O(2) VN(1,2)],[O(3) VN(1,3)],'LineWidth',2,'Color',.7*[1 1 1]);
        line([O(1) VN(2,1)],[O(2) VN(2,2)],[O(3) VN(2,3)],'LineWidth',2,'Color',.7*[1 1 1]);
        line([O(1) VN(3,1)],[O(2) VN(3,2)],[O(3) VN(3,3)],'LineWidth',2,'Color',.7*[1 1 1]);
        line([O(1) VN(4,1)],[O(2) VN(4,2)],[O(3) VN(4,3)],'LineWidth',2,'Color',.7*[1 1 1]);
      end
    end
    
    function obj = play(obj)
      global APPLICATION;
      tic;
      InitFrame = obj.CurrentFrame;
			fRate = ceil(50/obj.Frequency);
      for f = obj.CurrentFrame :fRate: obj.NumFrames
				obj = obj.seek(f);
				if APPLICATION.state ~= APPLICATION.RUNNING
					APPLICATION.state = APPLICATION.PAUSED;
					obj.CurrentFrame = f;
					return;
				end
				set(APPLICATION.handles.slider1,'value',f/obj.NumFrames);
				set(APPLICATION.handles.frame,'string',num2str(f));

        drawnow;
      end
      		
			set(APPLICATION.handles.slider1,'value',1/obj.NumFrames);
			set(APPLICATION.handles.frame,'string',num2str(1));
			APPLICATION.state = APPLICATION.STOPPED;
			obj.seek(1);
    end
      
    function obj = updateView(obj, nv)
      if strcmp(get(obj.Views{nv}.figure1,'visible'),'off')
        return;
			end
			
			% video
			im = obj.InputDataAccess{nv}.getFrame(obj.CurrentFrame);
			if obj.MirrorSymmetric(nv)	
				im = obj.VideoFeatures{2}.process(im);
				im = obj.VideoFeatures{2}.normalize(im);
			else
				im = obj.VideoFeatures{1}.process(im);
				im = obj.VideoFeatures{1}.normalize(im);
			end
			
			% masks
			if obj.Views{nv}.mask==2
				bboxmask = obj.MaskData{nv,obj.CurrentFrame};
				
				im(:,:,1) = im(:,:,1) .* uint16(bboxmask);
        im(:,:,2) = im(:,:,2) .* uint16(bboxmask);
        im(:,:,3) = im(:,:,3) .* uint16(bboxmask);
			elseif obj.Views{nv}.mask==3
				% if test set then load the mask (this is very inneficient so
				% computing is easier for training subjects
				if any(obj.Sequences{1}.Subject == [2 3 4 10]) || obj.Sequences{1}.Action == 1
					bboxmask = obj.BBMaskData{nv,obj.CurrentFrame};
				else
					if obj.MirrorSymmetric(nv)
						if isempty(obj.PoseData{6,nv})
							Pose = H36MComputeFeatures(obj.Sequences{nv},obj.PoseFeatures(6));
							obj.PoseData{6,nv} = Pose{1};
						end
						bboxmask = obj.Masks{obj.Views{nv}.mask}.process(obj.PoseData{6,nv}(obj.CurrentFrame,:),[],obj.Cameras{nv});
						bboxmask = obj.Masks{obj.Views{nv}.mask}.normalize(bboxmask);
					else
						if isempty(obj.PoseData{3,nv})
							Pose = H36MComputeFeatures(obj.Sequences{nv},obj.PoseFeatures(3));
							obj.PoseData{3,nv} = Pose{1};
						end
						bboxmask = obj.Masks{obj.Views{nv}.mask}.process(obj.PoseData{3,nv}(obj.CurrentFrame,:),[],obj.Cameras{nv});
						bboxmask = obj.Masks{obj.Views{nv}.mask}.normalize(bboxmask);
					end
				end
        im(:,:,1) = im(:,:,1) .* uint16(bboxmask);
        im(:,:,2) = im(:,:,2) .* uint16(bboxmask);
        im(:,:,3) = im(:,:,3) .* uint16(bboxmask);
      end
      
      if ~isfield(obj.Views{nv},'handleim')
        figure(obj.Views{nv}.figure1);
        obj.Views{nv}.handleim = image(im,'CDataMapping','direct'); 
        [h w ~] = size(im);
        axis equal; xlim([1 h]); ylim([1 w]); axis off;
      else
        set(obj.Views{nv}.handleim,'cdata',im);
			end
      
			% pose
      switch obj.PoseFeatures{obj.ShowPose(nv)}.FeatureName
        case 'NoPose'
          if ~isempty(obj.PoseHandles{nv})
            delete(obj.PoseHandles{nv}(2:end));
            obj.PoseHandles{nv}= [];
          end
          
        case 'D3_Positions'
					if obj.MirrorSymmetric(nv)
						P3D = obj.PoseData{obj.ShowPose(nv),nv}(obj.CurrentFrame,:);
					else
						P3D = obj.PoseData{obj.ShowPose(nv),nv}(obj.CurrentFrame,:);	
					end
					
					Skel = obj.Skel{obj.ShowPose(nv)};
					
          P2D = obj.Cameras{obj.Views{nv}.camera}.project(P3D);
          P2D = P2D * obj.Resize; 

          if isempty(obj.PoseHandles{nv})
            figure(obj.Views{nv}.figure1); hold on; 
            obj.PoseHandles{nv} = show2DPose(P2D, Skel);
            axis off;
          else
            update2DPose(obj.PoseHandles{nv},P2D, Skel);
          end
          
        case 'D3_Angles'
          A3D = obj.PoseData{obj.ShowPose(nv),nv}(obj.CurrentFrame,:);
          P3D = skel2xyz(obj.Skeletons{1},A3D);
          
          P2D = obj.Cameras{obj.Views{nv}.camera}.project(P3D);
          P2D = P2D * obj.Resize; 

          if isempty(obj.PoseHandles{nv})
            figure(obj.Views{nv}.figure1); hold on; 
            obj.PoseHandles{nv} = show2DPose(P2D, obj.Skel{obj.ShowPose(nv)});
            axis off;
          else
            update2DPose(obj.PoseHandles{nv},P2D, obj.Skel{obj.ShowPose(nv)});
					end
          
        case 'D2_Positions'
					if obj.MirrorSymmetric(nv)
						P2D = obj.PoseData{obj.ShowPose(nv),nv}(obj.CurrentFrame,:);
					else
						P2D = obj.PoseData{obj.ShowPose(nv),nv}(obj.CurrentFrame,:);
					end
          
          P2D = P2D * obj.Resize;
          if isempty(obj.PoseHandles{nv})
            figure(obj.Views{nv}.figure1); hold on; 
            obj.PoseHandles{nv} = show2DPose(P2D, obj.Skel{obj.ShowPose(nv)});
            axis off;
          else
            update2DPose(obj.PoseHandles{nv},P2D, obj.Skel{obj.ShowPose(nv)});
          end
          
      end
    end
    
    function obj = seek(obj,f)
      obj.CurrentFrame = f;
      
      for i = 1: obj.NumViews
        obj = obj.updateView(i);
      end
      obj = obj.updateScene();
      obj = obj.updateDepth();
		end
		
		function obj = setMirrorSymmetric(obj,nv,v)
			obj.MirrorSymmetric(nv) = v;
			if obj.ShowPose(nv)>1 && obj.ShowPose(nv) > 4
				obj = obj.setPose(nv, obj.ShowPose(nv)-3);
			elseif obj.ShowPose(nv)>1
				obj = obj.setPose(nv, obj.ShowPose(nv)+3);
			end
			
			if obj.Views{nv}.mask == 2
				obj.MaskFlip(nv) = ~obj.MaskFlip(nv);
				for i = 1: length(obj.MaskData)
					% this is a speedup
					obj.MaskData{nv,i} = fliplr(obj.MaskData{nv,i});
				end
			end
			
			if obj.Views{nv}.mask == 3 && (any(obj.Sequences{1}.Subject == [2 3 4 10]) || obj.Sequences{1}.Action == 1)
				obj.MaskFlip(nv) = ~obj.MaskFlip(nv);
				for i = 1: length(obj.BBMaskData)
					% this is a speedup
					obj.BBMaskData{nv,i} = fliplr(obj.BBMaskData{nv,i});
				end
			end
			obj.updateView(nv);
		end
    
    function obj = setNumViews(obj, nv)
      obj.NumViews = nv;
    end
    
    function obj = setView(obj, numview, handles)
      obj.Views{numview} = handles;
    end
    
    function obj = setSceneView(obj,handles)
      obj.SceneHandles = handles;
    end
    
    function obj = setDepthView(obj, handles)
      obj.DepthHandles = handles;
    end
    
    function obj = updateDepth(obj)
      if strcmp(get(obj.DepthHandles,'visible'),'off')
        return;
			end
      
			if ~isempty(obj.DepthData)
				set(obj.DepthHandles,'CData',ceil(100*obj.DepthData{obj.CurrentFrame}));
			end
    end
    
    function obj = updateScene(obj)
      if strcmp(get(obj.SceneHandles.figure1,'visible'),'off')
        return;
			end
			
			if isempty(obj.PoseData)
				i = 3; c=1;
				RawPose = H36MRawPoseFeature();
				if obj.PoseFeatures{i}.exist(obj.Sequences{c})==0
					process = true;
					RawPoseDataAccess{i} = RawPose.serializer(obj.Sequences{c});
				else
					process = false;
					PoseDataAccess{i} = obj.PoseFeatures{i}.serializer(obj.Sequences{c});
				end
						
				for f = 1: obj.NumFrames
					if process
						[pose RawPoseDataAccess{i}] = RawPoseDataAccess{i}.getFrame(f); 
						pose = obj.PoseFeatures{i}.process(pose,obj.Subject,obj.Cameras{c});
					else
						[pose PoseDataAccess{i}] = PoseDataAccess{i}.getFrame(f); 
						pose = pose';
					end
					obj.PoseData{i,c}(f,:) = obj.PoseFeatures{i}.normalize(pose,obj.Subject,obj.Cameras{c});
				end
			end
			
			if isempty(obj.PoseData{3,1})
				Pose = H36MComputeFeatures(obj.Sequences{1},obj.PoseFeatures(3));
				obj.PoseData{3,1} = Pose{1};
			end
			
      if ~isempty(obj.ScenePoseHandles)
        updatePose(obj.ScenePoseHandles,obj.PoseData{3}(obj.CurrentFrame,:),obj.Subject.getPosSkel());
      else
        figure(obj.SceneHandles.figure1);
        obj.showCameras();
        xlim(obj.SceneHandles.axes1,obj.XLimits); ylim(obj.SceneHandles.axes1,obj.YLimits); zlim(obj.SceneHandles.axes1,obj.ZLimits);
        axis equal; grid on; hold on;
        obj.ScenePoseHandles = showPose(obj.PoseData{3}(obj.CurrentFrame,:),obj.Subject.getPosSkel());
      end
		end
    
		function [obj exists] = loadTOF(obj)
			
			if isempty(obj.DepthData)
				disp('Loading TOF Data...');
				hf = waitbar(0,sprintf('Loading...'),'WindowStyle','modal');
				TOFDataAccess = H36MTOFDataAccess(obj.Subject.Number,obj.Action,obj.SubAction);
				exists = TOFDataAccess.exist();
				if ~exists
					close(hf);
					return;
				end

				minimum = 1; maximum = 0;
				for i = 1: obj.NumFrames
					obj.DepthData{i} = TOFDataAccess.getFrame(i);
					MM = max(max(obj.DepthData{i}));

					if maximum < MM
						maximum = MM;
					end
					mm = min(min(obj.DepthData{i}));
					if minimum > mm
						minimum = mm;
					end
					f = (maximum-minimum);
				end

				for i = 1: obj.NumFrames
					obj.DepthData{i} = obj.DepthData{i}/f;
				end
				close(hf);
			else
				exists = true;
			end
		end
		
    function obj = setCamera(obj, nv, camera)
      obj.Views{nv}.camera = camera;
      obj.Views{nv}.mask = 1;
      for i = 1: size(obj.MaskData,2)
        obj.MaskData{nv,i} = [];
			end
    end
        
    function obj = setMask(obj, nv, nummask)
      obj.Views{nv}.mask = nummask;
			      
			if nummask == 2 && isempty(obj.MaskData{nv,1})
				hf = waitbar(0,sprintf('Loading BS Mask...'),'WindowStyle','modal');
% 				MaskDataAccess = obj.Masks{nummask}.serializer(obj.Sequences{nv});
				load([obj.Sequences{nv}.getPath() filesep obj.Masks{nummask}.FeaturePath obj.Sequences{nv}.getName '.mat'],'Masks');
				
				for i = 1: obj.NumFrames
					waitbar(i/obj.NumFrames);

% 					[M MaskDataAccess] = MaskDataAccess.getFrame(i);
					M = Masks{i};
					if obj.MirrorSymmetric(nv)
						bboxmask = obj.Masks{obj.Views{nv}.mask+2}.normalize(M);
						obj.MaskFlip(nv) = true;
					else
						bboxmask = obj.Masks{obj.Views{nv}.mask}.normalize(M);
						obj.MaskFlip(nv) = false;
					end
					obj.MaskData{nv,i} = logical(bboxmask(:,:,1));
				end
				clear('Masks','var');
				
				close(hf);
			elseif nummask == 2
				hf = waitbar(0,sprintf('Loading BS Mask...'),'WindowStyle','modal');
				if obj.MirrorSymmetric(nv) ~= obj.MaskFlip(nv)
					obj.MaskFlip(nv) = ~obj.MaskFlip(nv);
					for i = 1: length(obj.MaskData)
						% this is a speedup
						obj.MaskData{nv,i} = fliplr(obj.MaskData{nv,i});
					end
				end
				
				close(hf);
			end
			
			if nummask == 3 && isempty(obj.BBMaskData{nv,1}) && (any(obj.Sequences{1}.Subject == [2 3 4 10]) || obj.Sequences{1}.Action == 1)
				hf = waitbar(0,sprintf('Loading BB Mask...'),'WindowStyle','modal');
% 				MaskDataAccess = obj.Masks{nummask}.serializer(obj.Sequences{nv});
				if ~exist([obj.Sequences{nv}.getPath() filesep obj.Masks{nummask}.FeaturePath obj.Sequences{nv}.getName '.mat'],'file')
					error('Please download BB mask file for this test sequence to continue!');
				end
				load([obj.Sequences{nv}.getPath() filesep obj.Masks{nummask}.FeaturePath obj.Sequences{nv}.getName '.mat'],'Masks');
				
				for i = 1: obj.NumFrames
					waitbar(i/obj.NumFrames);

% 					[M MaskDataAccess] = MaskDataAccess.getFrame(i);
					M = Masks{i};
					if obj.MirrorSymmetric(nv)
						bboxmask = obj.Masks{obj.Views{nv}.mask+2}.normalize(M);
						obj.MaskFlip(nv) = true;
					else
						bboxmask = obj.Masks{obj.Views{nv}.mask}.normalize(M);
						obj.MaskFlip(nv) = false;
					end
					obj.BBMaskData{nv,i} = logical(bboxmask(:,:,1));
				end
				clear('Masks','var');
				
				close(hf);
			elseif nummask == 3 && (any(obj.Sequences{1}.Subject == [2 3 4 10]) || obj.Sequences{1}.Action == 1)
				hf = waitbar(0,sprintf('Loading BB Mask...'),'WindowStyle','modal');
				if obj.MirrorSymmetric(nv) ~= obj.MaskFlip(nv)
					obj.MaskFlip(nv) = ~obj.MaskFlip(nv);
					for i = 1: length(obj.BBMaskData)
						% this is a speedup
						obj.BBMaskData{nv,i} = fliplr(obj.BBMaskData{nv,i});
					end
				end
				
				close(hf);
			end
      
		end
    
		function obj = loadCamera(obj, c)
			tic;				
			if isempty(obj.InputDataAccess) || isempty(obj.InputDataAccess{c})
				hf = waitbar(0,sprintf('Loading Video...'),'WindowStyle','modal');
				disp('Loading Camera...');
				VF{c} = H36MRGBVideoFeature();
				obj.InputDataAccess{c} = VF{c}.serializer(obj.Sequences{c});
				toc;

				close(hf);
			end
		end
		
    function obj = setPose(obj, nv, v)
			obj.ShowPose(nv) = v;
			
			hf = waitbar(0,sprintf('Loading Pose...'),'WindowStyle','modal');
			if (v~=1) &&(isempty(obj.PoseData{v,nv}))
				Pose = H36MComputeFeatures(obj.Sequences{nv},obj.PoseFeatures(v));
				obj.PoseData{v,nv} = Pose{1};
			end
			
			close(hf);
		end
		
		function obj = clear(obj)
			for c = 1: 4
				if ~isempty(obj.InputDataAccess) && ~isempty(obj.InputDataAccess{c})
					try
						obj.InputDataAccess{c}.delete;
					catch e
						e
					end
				end
			end
		end
			
  end
  
end