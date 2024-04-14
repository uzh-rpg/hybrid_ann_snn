% H36MDataBase object is the main source of information about the data
% Any type of information about subjects, actions, trials etc can be obtained
% here.

classdef H36MDataBase
  properties
    mapping; 
    database;
    toffiles;
    frames;
    exp_dir;
		model_dir;
    w0;
    
    subjects;
    actions;
		actionnames;
    cameras;
    subactions;
    
    train_subjects;
    val_subjects;
    test_subjects;
    
    skel_angles;
    skel_pos;
		subject_measurements;
		
		renders;
		experiment;
  end
  
  methods(Access=private)
    function obj = H36MDataBase()
      disp('Init DataBase!...');
			if ~exist('H36M.conf','file')
				disp('Please input the path to the data.');
				if usejava('jvm')
					expdir = uigetdir('.','Please input the path to the data.');
				else
					expdir = input('Data path:', 's');
				end
				hf = fopen('H36M.conf','w+');
				fprintf(hf,'%s/',expdir);
				fclose(hf);
				
				disp('Please config file path.');
				disp('Please config file path.');
				if usejava('jvm')
					metadir = uigetdir('*.xml','Please input the directory to the config file (metadata.xml). It should be in your code archive.');
				else
					metadir = input('Please input the directory to the config file (metadata.xml). It should be in your code archive. ', 's');
				end
				copyfile([metadir filesep 'metadata.xml'],expdir);
				copyfile([metadir filesep 'metadata_renders.xml'],expdir);
			end
			hf = fopen('H36M.conf','r');
			obj.exp_dir = fscanf(hf,'%s');
			fclose(hf);
			
			if iscluster()
				obj.model_dir = obj.exp_dir;%'/home/catalin/H36MData/'
			else
				obj.model_dir = obj.exp_dir;
			end
			
			if ~exist([obj.exp_dir 'metadata.xml'],'file')
				error(['Installation not complete. Please copy metadata.xml file to ' obj.exp_dir]);
			end
      xmlobj = xml_read([obj.exp_dir 'metadata.xml']);
			
      obj.frames = xmlobj.frames;
      obj.mapping = xmlobj.mapping;
      obj.database.cameras = xmlobj.dbcameras;
      obj.toffiles = xmlobj.toffiles;
      obj.w0 = xmlobj.w0;
      obj.skel_angles = xmlobj.skel_angles;
			obj.skel_angles.tree = xmlobj.skel_angles.tree'; % for some reason the xml writer gets things transposed
			obj.skel_pos = xmlobj.skel_pos;
			obj.actionnames = xmlobj.actionnames;
			obj.subject_measurements = xmlobj.subject_measurements;
			
      for i = 1 : 11
        obj.subjects{i} = [];
      end
      
      obj.actions = 1:16;
      obj.cameras = cell(11,4);
      obj.subactions = 1:2;
      
      obj.train_subjects = xmlobj.train_subjects;
      obj.val_subjects = xmlobj.val_subjects;
      obj.test_subjects = xmlobj.test_subjects;
			if exist([obj.exp_dir 'metadata_renders.xml'],'file')
				obj.renders = xml_read([obj.exp_dir 'metadata_renders.xml']);
			end
		end
	end
	
  
  methods(Static)
    function obj = instance()
       persistent uniqueInstance
       if isempty(uniqueInstance)
          obj = H36MDataBase();
          uniqueInstance = obj;
       else
          obj = uniqueInstance;
       end
    end
  end
  
  methods		
    function [cam obj] = getCamera(obj, s, c)		
			if s == 0
				cam = H36MCamera(obj, s, 1);
				return;
			end
			
      if isempty(obj.cameras{s,c})
        obj.cameras{s,c} = H36MCamera(obj, s, c);
      end
      cam = obj.cameras{s,c};
		end
    
		function res = getResolution(obj,s,c)
			if s == 0
				res = obj.mapping{35,1+2}(c,:);
			else
				res = obj.mapping{35,s+2}(c,:);
			end
		end
		
    function subj = getSubject(obj, s)			
			if isempty(obj.subjects{s})
				obj.subjects{s} = H36MSubject(obj, s);
			end
      subj = obj.subjects{s};
    end
       
    function skel = getAnglesSkel(obj,i)
			if i == 0
				% FIXME renders subject
				i = 1;
			end
      skel = GetSkelMeasurements(obj.skel_angles,obj.subject_measurements(i,:),false);
		end
    
		function meas = getSubjectMeasurements(obj,i)
			meas = obj.subject_measurements(i,:);
		end
		
    function skel = getPosSkel(obj)
      skel = obj.skel_pos;
    end
    
    function name = getSubjectName(obj, subject)
			if subject == 0
				name = 'Renders';
			else
				name = obj.mapping{1,subject+2};
			end
		end
    
    function name = getSubjectAlias(obj, subject)
      name = obj.mapping{1,subject+2};
    end

    function filename = getFileName(obj, subject, action, subaction, camera)
			if subject==0
				filename = obj.renders.filenames{action};
			else
				filename = obj.mapping{action*2+subaction-1,subject+2};

				if exist('camera','var')
					filename = [filename '.' num2str(obj.database.cameras.index2id{camera})];
				end
			end
    end
    
    function filename = getTOFFileName(obj, subject, action, subaction)
      filename = ['TOF' filesep 'TOF' num2str(obj.toffiles{action*2+subaction-1,subject+2}) '.mat'];
		end
    
		function subj = getUniversalSubject(obj)
			subj = obj.getSubject(1);
		end
		
		function skel = getUniversalAnglesSkel(obj)
			skel = obj.getUniversalSubject.getAnglesSkel;
		end
		
		function skel = getUniversalPosSkel(obj)
			skel = obj.getUniversalSubject.getPosSkel;
    end
    
    function skel = getUniversal2DPosSkel(obj)
			skel = obj.getUniversalSubject.get2DPosSkel;
    end
    
    function trial = getTrialName(obj, Subject, Action, SubAction)
      trial = obj.mapping{Action*2+SubAction-1,Subject+2};
    end
    
    function cam_name = getCameraName(obj, camera)
      cam_name = obj.database.cameras.index2id{camera};
		end
    
		function obj = setExperiment(obj, exp)
			if isempty(obj.experiment)
				obj.experiment = exp;
			else
				error('Only one experiment at a time! Please clear the instance first!');
			end
		end
		
		function exp = getExperiment(obj)
			exp = obj.experiment;
		end
		
    function cam = loadCamera(obj, subject, camera)
			w1 = zeros(1,15);
			start = 6*((camera-1)*11 + (subject-1)) + 1; 
			w1(1:6) = obj.w0(start:start+5);
			w1(7:end) = obj.w0((265+(camera-1)*9):(264+camera*9));
      cam = w1;
		end
    
    function num_frames = getNumFrames(obj, subject, action, subaction)
			if subject==0
				num_frames = obj.renders.num_frames(action);
			else
				num_frames = min(min(obj.frames{subject,action*2+subaction-1}(1:4)),floor(obj.frames{subject,action*2+subaction-1}(5)/4));
			end
    end
  end
end