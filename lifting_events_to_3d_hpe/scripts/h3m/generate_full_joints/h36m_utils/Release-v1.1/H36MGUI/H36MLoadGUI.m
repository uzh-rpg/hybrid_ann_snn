function varargout = H36MLoadGUI(varargin)
% H36MLOADGUI MATLAB code for H36MLoadGUI.fig
%      H36MLOADGUI, by itself, creates a new H36MLOADGUI or raises the existing
%      singleton*.
%
%      H = H36MLOADGUI returns the handle to a new H36MLOADGUI or the handle to
%      the existing singleton*.
%
%      H36MLOADGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in H36MLOADGUI.M with the given input arguments.
%
%      H36MLOADGUI('Property','Value',...) creates a new H36MLOADGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before H36MLoadGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to H36MLoadGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help H36MLoadGUI

% Last Modified by GUIDE v2.5 06-Nov-2012 16:57:20

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @H36MLoadGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @H36MLoadGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before H36MLoadGUI is made visible.
function H36MLoadGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to H36MLoadGUI (see VARARGIN)

% Choose default command line output for H36MLoadGUI
handles.output = hObject;
global APPLICATION;
APPLICATION.handle_LoadGUI = hObject;

% Update handles structure
guidata(hObject, handles);
set(hObject,'WindowStyle','modal');

% UIWAIT makes H36MLoadGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = H36MLoadGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1
global APPLICATION;
subj = get(hObject,'value');
% APPLICATION = setSequence(APPLICATION,subj,APPLICATION.action,APPLICATION.subaction);

i = 1;
for a = 1 : 16
  for sa = 1: 2
    s{i} = APPLICATION.database.getFileName(APPLICATION.subject, a, sa);
    i = i + 1;
  end
end
set(handles.popupmenu2,'string',s);
set(handles.popupmenu2,'value',1);

% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

global APPLICATION;
for i = 1 : 11
  s{i} = APPLICATION.database.getSubject(i).Name;
end
set(hObject,'string',s);


% --- Executes on selection change in popupmenu2.
function popupmenu2_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu2
global APPLICATION;
v = get(hObject,'value');
a = floor(v./2) + 1;
sa = rem(v,2);
% APPLICATION = setSequence(APPLICATION,APPLICATION.subject, a, sa);

% --- Executes during object creation, after setting all properties.
function popupmenu2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
global APPLICATION;
i = 1;
for a = 1 : 16
  for sa = 1: 2
		filename = APPLICATION.database.getFileName(APPLICATION.subject, a, sa);
		s{i} = filename;
		i = i + 1;
  end
end

set(hObject,'string',s);
set(hObject,'value',(APPLICATION.action-1)*2+APPLICATION.subaction);

% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
disp('Loading...');
global APPLICATION; 
% disp(APPLICATION.database.getSubject(APPLICATION.subject).Name);
APPLICATION.subject=get(handles.popupmenu1,'value');
numaction = get(handles.popupmenu2,'value');
APPLICATION.action=ceil(numaction/2);
APPLICATION.subaction=numaction - (APPLICATION.action-1)*2;

disp(APPLICATION.database.getFileName(APPLICATION.subject,APPLICATION.action,APPLICATION.subaction));
% check files
for i = 1: 4
	vf(i) = exist([APPLICATION.database.exp_dir APPLICATION.database.getSubjectName(APPLICATION.subject) filesep  'Videos' filesep APPLICATION.database.getFileName(APPLICATION.subject,APPLICATION.action,APPLICATION.subaction,i) '.mp4'],'file');
	mf(i) = exist([APPLICATION.database.exp_dir  APPLICATION.database.getSubjectName(APPLICATION.subject) filesep 'MySegmentsMat' filesep 'ground_truth_bs' filesep APPLICATION.database.getFileName(APPLICATION.subject,APPLICATION.action,APPLICATION.subaction,i) '.mat'],'file');
end
if ~all(vf) || ~all(mf)
	if ~all(vf)
		ed = errordlg('Video files missing please download.','Error');
		set(ed, 'WindowStyle', 'modal');
		uiwait(ed);
	else  ~all(mf)
		ed = errordlg('Mask files missing please download.','Error');
		set(ed, 'WindowStyle', 'modal');
		uiwait(ed);
	end
	return;
end
% set(handles.figure1,'visible','off');

APPLICATION.state = APPLICATION.STOPPED;


if isempty(APPLICATION.views)
  APPLICATION.sceneview_handle = H36MSceneViewer('visible','off');
  a = get(0,'screensize');
  w = a(3);
	h=a(4);
	%pos = [0 h; 550 h; 0 0; 550 0];
	pos = [0 550; 550 550; 0 0; 550 0];
  for i = 1:4
		matver = version('-release');
		if strcmp(matver(1:4),'2012')
			APPLICATION.views{i} = H36MViewGUI_2012(i);
		else
			APPLICATION.views{i} = H36MViewGUI(i);
		end
    
    b = get(APPLICATION.views{i}, 'Position'); 

    set(APPLICATION.views{i},'Position',[pos(i,:) b(3:4)]);
  end
  APPLICATION.visualiser = APPLICATION.visualiser.setNumViews(4);
  
  set(APPLICATION.handles.stop,'enable','on');
  set(APPLICATION.handles.fwd,'enable','on');
  set(APPLICATION.handles.play,'enable','on');
  set(APPLICATION.handles.oneview,'enable','on');
  set(APPLICATION.handles.twoview,'enable','on');
  set(APPLICATION.handles.threeview,'enable','on');
  set(APPLICATION.handles.fourview,'enable','on');
  set(APPLICATION.handles.slider1,'enable','on');
	
%   set(APPLICATION.handles.graph,'enable','on');
  set(APPLICATION.handles.depth,'enable','on');
  set(APPLICATION.handles.slider1,'value',0);
end
set(APPLICATION.handles.slider1,'value',1);
db = H36MDataBase.instance();

APPLICATION.visualiser = APPLICATION.visualiser.load(APPLICATION.subject,APPLICATION.action,APPLICATION.subaction);
if ~isfield(APPLICATION,'depth_handle')
  APPLICATION.depth_handle = figure('visible','off');
	set(APPLICATION.depth_handle,'CloseRequestFcn',@depth_closefcn);
	depthax_handle = image; axis ij; axis off;
	APPLICATION.visualiser = APPLICATION.visualiser.setDepthView(depthax_handle);
else
	set(APPLICATION.depth_handle,'visible','off');
end

if any(APPLICATION.subject==db.test_subjects)||APPLICATION.action == 1
	set(APPLICATION.handles.scene,'enable','off');
	set(APPLICATION.handles.depth,'enable','off');
else
	set(APPLICATION.handles.scene,'enable','on');
	set(APPLICATION.handles.depth,'enable','on');
end

for i = 1: length(APPLICATION.visualiser.Views)
  set(APPLICATION.views{i},'visible','off');
	
	children = get(APPLICATION.views{i},'Children');
	for j = 2:3
		set(children(j),'Value',1);
	end
	set(children(1),'Value',0);
	
  APPLICATION.visualiser=APPLICATION.visualiser.updateView(i);
end

% APPLICATION.visualiser=APPLICATION.visualiser.updateScene();
set(APPLICATION.handle_mainwindow,'visible','on');
set(APPLICATION.handle_LoadGUI,'visible','off');
set(APPLICATION.sceneview_handle,'visible','off');
setcurrentframe(1);

% function APPLICATION = setSequence(APPLICATION, subject, action, subaction)
% APPLICATION.tmp_subject = subject;
% APPLICATION.tmp_action = action;
% APPLICATION.tmp_subaction = subaction;

function depth_closefcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
% delete(hObject);
set(hObject,'visible','off');

% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
% delete(hObject);
global APPLICATION;
set(hObject,'visible','off');
try
  set(APPLICATION.handle_mainwindow,'visible','on');
catch e
end

function setcurrentframe(f)
global APPLICATION;
set(APPLICATION.handles.slider1,'value',(max(f,1)-1)/(APPLICATION.visualiser.NumFrames-1));
set(APPLICATION.handles.frame,'string',num2str(max(f,1)));
APPLICATION.visualiser = APPLICATION.visualiser.seek(max(f,1));


