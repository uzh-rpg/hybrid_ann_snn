function varargout = H36MViewGUI_2012(varargin)
% H36MVIEWGUI MATLAB code for H36MViewGUI.fig
%      H36MVIEWGUI, by itself, creates a new H36MVIEWGUI or raises the existing
%      singleton*.
%
%      H = H36MVIEWGUI returns the handle to a new H36MVIEWGUI or the handle to
%      the existing singleton*.
%
%      H36MVIEWGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in H36MVIEWGUI.M with the given input arguments.
%
%      H36MVIEWGUI('Property','Value',...) creates a new H36MVIEWGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before H36MViewGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to H36MViewGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help H36MViewGUI

% Last Modified by GUIDE v2.5 07-Mar-2013 17:06:57

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ... 
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @H36MViewGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @H36MViewGUI_OutputFcn, ...
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


% --- Executes just before H36MViewGUI is made visible.
function H36MViewGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to H36MViewGUI (see VARARGIN)

% Choose default command line output for H36MViewGUI
handles.output = hObject;

% Update handles structure
handles.numview = varargin{1};
guidata(hObject, handles);

set(hObject,'name',['View ' num2str(handles.numview)]);
global APPLICATION;
handles.camera = handles.numview;
APPLICATION.visualiser = APPLICATION.visualiser.setView(handles.numview, handles);
APPLICATION.visualiser = APPLICATION.visualiser.setMask(handles.numview, 1);
set(handles.cameras,'value',handles.numview);
set(hObject,'Renderer','OpenGL');
% set(hObject,'Renderer','zbuffer')
% UIWAIT makes H36MViewGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = H36MViewGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in cameras.
function cameras_Callback(hObject, eventdata, handles)
% hObject    handle to cameras (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns cameras contents as cell array
%        contents{get(hObject,'Value')} returns selected item from cameras
disp('Change camera');
% disp(get(hObject,'Value'));

disp(handles.numview);
global APPLICATION;
APPLICATION.visualiser = APPLICATION.visualiser.setCamera(handles.numview, get(hObject,'value'));
set(handles.mask,'value',1);

% --- Executes during object creation, after setting all properties.
function cameras_CreateFcn(hObject, eventdata, handles)
% hObject    handle to cameras (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

for i = 1: 4
  s{i} = ['RGB ' num2str(i)];
end
set(hObject,'string',s);

% --- Executes on selection change in mask.
function mask_Callback(hObject, eventdata, handles)
% hObject    handle to mask (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns mask contents as cell array
%        contents{get(hObject,'Value')} returns selected item from mask
disp('Change mask');
handles.numview
disp(get(hObject,'Value'));
global APPLICATION;
APPLICATION.visualiser = APPLICATION.visualiser.setMask(handles.numview, get(hObject,'value'));

% --- Executes during object creation, after setting all properties.
function mask_CreateFcn(hObject, eventdata, handles)
% hObject    handle to mask (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
global APPLICATION;
for i = 1: 3%length(APPLICATION.visualiser.Masks)
  s{i} = APPLICATION.visualiser.Masks{i}.FeatureName;
end
set(hObject,'string',s);
set(hObject,'value',1);

% masks = APPLICATION.visualiser.Masks;
% for i = 1: length(masks)
%   s{i} = masks{i}.FeatureName;
% end
% set(hObject,'string',s);

% --- Executes on button press in pose.
function pose_Callback(hObject, eventdata, handles)
% hObject    handle to pose (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of pose
v = get(hObject,'value');
global APPLICATION;

db = H36MDataBase.instance();
if any(APPLICATION.subject == [db.test_subjects])
	ed = errordlg('This is a test subject so poses are not available!','Error');
	set(ed, 'WindowStyle', 'modal');
	uiwait(ed);
	set(handles.pose,'value',1);
else
	if v==2 && APPLICATION.visualiser.MirrorSymmetric(handles.numview)

		ed = errordlg('Mirror symmetric transformation is not available for the angles representation.','Error');
		set(ed, 'WindowStyle', 'modal');
		uiwait(ed);
		set(handles.mirror2,'value',0);
		APPLICATION.visualiser= APPLICATION.visualiser.setMirrorSymmetric(handles.numview,0);
	end

	APPLICATION.visualiser = APPLICATION.visualiser.setPose(handles.numview,v);
end

% --- Executes during object creation, after setting all properties.
function axes_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes
% set(hObject,'visibility','off');


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(hObject,'visible','off');
% Hint: delete(hObject) closes the figure
% delete(hObject);


% --- Executes during object creation, after setting all properties.
function pose_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pose (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
global APPLICATION;
for i = 1 : 4%length(APPLICATION.visualiser.PoseFeatures)
  s{i} = APPLICATION.visualiser.PoseFeatures{i}.FeatureName;
end
set(hObject,'string',s);


% --- Executes on button press in mirror2.
function mirror2_Callback(hObject, eventdata, handles)
% hObject    handle to mirror2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of mirror2
global APPLICATION;
v = get(handles.mirror2,'value');
if APPLICATION.visualiser.ShowPose(handles.numview) == 2
	ed = errordlg('Mirror symmetric transformation is not available for the angles representation.','Error');
	set(ed, 'WindowStyle', 'modal');
	uiwait(ed);
	set(handles.mirror2,'value',0);
else
	APPLICATION.visualiser = APPLICATION.visualiser.setMirrorSymmetric(handles.numview,v);
end


% --- Executes on button press in mirror.
function mirror_Callback(hObject, eventdata, handles)
% hObject    handle to mirror (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of mirror
global APPLICATION;
v = get(handles.mirror,'value');
if APPLICATION.visualiser.ShowPose(handles.numview) == 2
	ed = errordlg('Mirror symmetric transformation is not available for the angles representation.','Error');
	set(ed, 'WindowStyle', 'modal');
	uiwait(ed);
	set(handles.mirror,'value',0);
else
	APPLICATION.visualiser = APPLICATION.visualiser.setMirrorSymmetric(handles.numview,v);
end
APPLICATION.visualiser = APPLICATION.visualiser.updateView(handles.numview);

