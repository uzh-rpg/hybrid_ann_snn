function varargout = H36MGui(varargin)
% H36MGUI MATLAB code for H36MGui.fig
%      H36MGUI, by itself, creates a new H36MGUI or raises the existing
%      singleton*.
%
%      H = H36MGUI returns the handle to a new H36MGUI or the handle to
%      the existing singleton*.
%
%      H36MGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in H36MGUI.M with the given input arguments.
%
%      H36MGUI('Property','Value',...) creates a new H36MGUI or raises
%      the existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before H36MGui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to H36MGui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help H36MGui

% Last Modified by GUIDE v2.5 13-Dec-2012 17:31:41


% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @H36MGui_OpeningFcn, ...
                   'gui_OutputFcn',  @H36MGui_OutputFcn, ...
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

% --- Executes just before H36MGui is made visible.
function H36MGui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to H36MGui (see VARARGIN)

% Choose default command line output for H36MGui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

addpaths;

initialize_gui(hObject, handles, false);

global APPLICATION;
APPLICATION.handle_loadgui = H36MLoadGUI('visible','off');% handles.figure1
APPLICATION.handle_mainwindow = handles.figure1;
set(APPLICATION.handle_loadgui,'visible','off');
set(handles.slider1,'enable','off');

% UIWAIT makes H36MGui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = H36MGui_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



% --------------------------------------------------------------------
function initialize_gui(fig_handle, handles, isreset)

global APPLICATION;
APPLICATION.INIT = 0;
APPLICATION.STOPPED = 1;
APPLICATION.PAUSED = 2;
APPLICATION.RUNNING = 3;
APPLICATION.RECORDING = 4;

APPLICATION.state = APPLICATION.INIT;
APPLICATION.database = H36MDataBase.instance();
APPLICATION.subject = 1;
APPLICATION.action = 13;
APPLICATION.subaction = 1;
APPLICATION.tmp_subject = 1;
APPLICATION.tmp_action = 13;
APPLICATION.tmp_subaction = 1;
APPLICATION.views = {};
APPLICATION.visualiser = H36MVideoVisualizer();

% Update handles structure
guidata(handles.figure1, handles);

set(handles.stop,'enable','off');
set(handles.fwd,'enable','off');

set(handles.play,'enable','off');

APPLICATION.handles = handles;

% 
% APPLICATION.views{1} = H36MViewGUI(1);
% APPLICATION.visualiser = APPLICATION.visualiser.setNumViews(1);


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
s = get(hObject,'value');
global APPLICATION;
if APPLICATION.state ~= APPLICATION.RUNNING
  setcurrentframe(ceil(s*APPLICATION.visualiser.NumFrames));
end
if APPLICATION.state == APPLICATION.STOPPED
	APPLICATION.state = APPLICATION.PAUSED;
end


% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% --- Executes on button press in play.
function play_Callback(hObject, eventdata, handles)
% hObject    handle to play (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global APPLICATION;
switch APPLICATION.state 
  case APPLICATION.RUNNING
    APPLICATION.state = APPLICATION.PAUSED;
    image_pic = im2double(imread('../H36MGUI/icons/media-playback-start.png'));
    set(hObject,'cdata',repmat(image_pic,[1 1 3]));
%     disp('Pausing');
		% turn off controls for windows
    
		for i = 1: length(APPLICATION.views)
			children = get(APPLICATION.views{i},'Children');
			for j = 1:length(children)-1
        if strcmp(get(children(j),'tag'),'cameras')
          set(children(j),'enable','inactive');
        else
          set(children(j),'enable','on');
        end
      end
		end
		set(handles.fwd,'enable','on');
		set(handles.oneview,'enable','on');
		set(handles.twoview,'enable','on');
		set(handles.threeview,'enable','on');
		set(handles.fourview,'enable','on');
		if ~any(APPLICATION.subject==[2 3 4 10])&&APPLICATION.action ~= 1
			set(handles.depth,'enable','on');
			set(handles.scene,'enable','on');
		end
		set(handles.eject,'enable','on');
		
  case {APPLICATION.PAUSED,APPLICATION.STOPPED}
		if (APPLICATION.STOPPED==APPLICATION.state)
			setcurrentframe(1);
		end
			
    APPLICATION.state = APPLICATION.RUNNING;
%     disp('Starting');
    image_pic = im2double(imread('../H36MGUI/icons/media-playback-pause.png'));
    set(hObject,'cdata',repmat(image_pic,[1 1 3]));
		% turn off controls for windows
		for i = 1: length(APPLICATION.views)
			children = get(APPLICATION.views{i},'Children');
			for j = 1:length(children)-1
				set(children(j),'enable','off');
			end
		end
		
		% turn of the toggle views as well as framerate buttons
		set(handles.fwd,'enable','off');
		set(handles.oneview,'enable','off');
		set(handles.twoview,'enable','off');
		set(handles.threeview,'enable','off');
		set(handles.fourview,'enable','off');
		set(handles.depth,'enable','off');
		set(handles.scene,'enable','off');
		set(handles.eject,'enable','off');
		
    APPLICATION.visualiser = APPLICATION.visualiser.play();

		set(handles.fwd,'enable','on');
		set(handles.oneview,'enable','on');
		set(handles.twoview,'enable','on');
		set(handles.threeview,'enable','on');
		set(handles.fourview,'enable','on');
		if ~any(APPLICATION.subject==[2 3 4 10])&&APPLICATION.action ~= 1
			set(handles.depth,'enable','on');
			set(handles.scene,'enable','on');
		end
		set(handles.eject,'enable','on');
		
    image_pic = im2double(imread('../H36MGUI/icons/media-playback-start.png'));
    set(hObject,'cdata',repmat(image_pic,[1 1 3]));
    
  otherwise
    error('err');
end
  
% --- Executes on button press in rec.
function rec_Callback(hObject, eventdata, handles)
% hObject    handle to rec (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% disp('Record');
global APPLICATION;
APPLICATION.state = APPLICATION.RECORDING;

% --- Executes on button press in stop.
function stop_Callback(hObject, eventdata, handles)
% hObject    handle to stop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% disp('Stop');
global APPLICATION;
APPLICATION.state = APPLICATION.STOPPED;

image_pic = im2double(imread('../H36MGUI/icons/media-playback-start.png'));
set(handles.play,'cdata',repmat(image_pic,[1 1 3]));
for i = 1: length(APPLICATION.views)
  children = get(APPLICATION.views{i},'Children');
  for j = 1:length(children)-1
    if strcmp(get(children(j),'tag'),'cameras')
      set(children(j),'enable','inactive');
    else
      set(children(j),'enable','on');
    end
  end
end
setcurrentframe(1)

% --- Executes on button press in fwd.
function fwd_Callback(hObject, eventdata, handles)
% hObject    handle to fwd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% disp('FWD');
global APPLICATION;
freq = APPLICATION.visualiser.Frequency;
switch freq
	case 50
		freq = 25;
	case 25
		freq = 10;
	case 10
		freq = 50;
end
set(handles.Freq,'String',num2str(freq));
APPLICATION.visualiser.Frequency = freq;

% --- Executes during object creation, after setting all properties.
function stop_CreateFcn(hObject, eventdata, handles)
% hObject    handle to stop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

image_pic = im2double(imread('../H36MGUI/icons/media-playback-stop.png'));
set(hObject,'cdata',repmat(image_pic,[1, 1, 3]));
set(hObject,'String','');

% --- Executes during object creation, after setting all properties.
function play_CreateFcn(hObject, eventdata, handles)
% hObject    handle to play (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

image_pic = im2double(imread('../H36MGUI/icons/media-playback-start.png'));
set(hObject,'cdata',repmat(image_pic,[1, 1, 3]))
set(hObject,'String','');

% --- Executes during object creation, after setting all properties.
function fwd_CreateFcn(hObject, eventdata, handles)
% hObject    handle to fwd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

image_pic = im2double(imread('../H36MGUI/icons/media-seek-forward.png'));
set(hObject,'cdata',image_pic)
set(hObject,'String','');

% --- Executes during object creation, after setting all properties.
function rec_CreateFcn(hObject, eventdata, handles)
% hObject    handle to rec (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
image_pic = im2double(imread('../H36MGUI/icons/media-record.png'));
set(hObject,'cdata',image_pic)
set(hObject,'String','');

% --- Executes on button press in eject.
function eject_Callback(hObject, eventdata, handles)
% hObject    handle to eject (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global APPLICATION;
set(APPLICATION.handle_loadgui,'visible','on');
% set(handles.stop,'enable','on');
% set(handles.fwd,'enable','on');
% set(handles.play,'enable','on');
% % set(handles.rec,'enable','on');
set(handles.figure1,'visible','off');

% --- Executes during object creation, after setting all properties.
function eject_CreateFcn(hObject, eventdata, handles)
% hObject    handle to eject (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
image_pic = im2double(imread('../H36MGUI/icons/media-eject.png'));
set(hObject,'cdata',repmat(image_pic,[1, 1, 3]))
set(hObject,'String','');


% --- Executes on button press in oneview.
function oneview_Callback(hObject, eventdata, handles)
% hObject    handle to oneview (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global APPLICATION;
nv = 1;
v = get(APPLICATION.views{nv},'visible');
switch v
  case 'off'
		APPLICATION.visualiser=APPLICATION.visualiser.loadCamera(nv);
    set(APPLICATION.views{nv},'visible','on');
		APPLICATION.visualiser.updateView(nv);
  case 'on'
    set(APPLICATION.views{nv},'visible','off');
end

% --- Executes on button press in twoview.
function twoview_Callback(hObject, eventdata, handles)
% hObject    handle to twoview (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global APPLICATION;
nv = 2;
v = get(APPLICATION.views{nv},'visible');
switch v
  case 'off'
    APPLICATION.visualiser=APPLICATION.visualiser.loadCamera(nv);
    set(APPLICATION.views{nv},'visible','on');
		APPLICATION.visualiser.updateView(nv);
  case 'on'
    set(APPLICATION.views{nv},'visible','off');
end

% --- Executes on button press in threeview.
function threeview_Callback(hObject, eventdata, handles)
% hObject    handle to threeview (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global APPLICATION;
nv = 3;
v = get(APPLICATION.views{nv},'visible');
switch v
  case 'off'
    APPLICATION.visualiser=APPLICATION.visualiser.loadCamera(nv);
    set(APPLICATION.views{nv},'visible','on');
		APPLICATION.visualiser.updateView(nv);
  case 'on'
    set(APPLICATION.views{nv},'visible','off');
end

% --- Executes on button press in fourview.
function fourview_Callback(hObject, eventdata, handles)
% hObject    handle to fourview (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global APPLICATION;
nv = 4;
v = get(APPLICATION.views{nv},'visible');
switch v
  case 'off'
		APPLICATION.visualiser=APPLICATION.visualiser.loadCamera(nv);
    set(APPLICATION.views{nv},'visible','on');
		APPLICATION.visualiser.updateView(nv);
  case 'on'
    set(APPLICATION.views{nv},'visible','off');
end


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
global APPLICATION;

for i = 1: length(APPLICATION.views)
	try
		delete(APPLICATION.views{i});
	catch e
	end
end

APPLICATION.visualiser.clear;
try
	delete(APPLICATION.sceneview_handle);
	delete(APPLICATION.depth_handle);
	delete(hObject);
catch e
end
close all;

clear('APPLICATION');
clear all;





function setcurrentframe(f)
global APPLICATION;
set(APPLICATION.handles.slider1,'value',(max(f,1)-1)/(APPLICATION.visualiser.NumFrames-1));
set(APPLICATION.handles.frame,'string',num2str(max(f,1)));
APPLICATION.visualiser = APPLICATION.visualiser.seek(max(f,1));


% --- Executes on button press in scene.
function scene_Callback(hObject, eventdata, handles)
% hObject    handle to scene (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global APPLICATION;
v = get(APPLICATION.sceneview_handle,'visible');
switch v
  case 'off'
    set(APPLICATION.sceneview_handle,'visible','on');
    APPLICATION.visualiser = APPLICATION.visualiser.updateScene();
  case 'on'
    set(APPLICATION.sceneview_handle,'visible','off');
end


% --- Executes during object creation, after setting all properties.
function oneview_CreateFcn(hObject, eventdata, handles)
% hObject    handle to oneview (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(hObject,'enable','off');

% --- Executes during object creation, after setting all properties.
function twoview_CreateFcn(hObject, eventdata, handles)
% hObject    handle to twoview (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(hObject,'enable','off');

% --- Executes during object creation, after setting all properties.
function threeview_CreateFcn(hObject, eventdata, handles)
% hObject    handle to threeview (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(hObject,'enable','off');

% --- Executes during object creation, after setting all properties.
function fourview_CreateFcn(hObject, eventdata, handles)
% hObject    handle to fourview (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(hObject,'enable','off');


% --- Executes on button press in graph.
function graph_Callback(hObject, eventdata, handles)
% hObject    handle to graph (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global APPLICATION;
v = get(APPLICATION.graphview_handle,'visible');
switch v
  case 'off'
    set(APPLICATION.graphview_handle,'visible','on');
%     APPLICATION.visualiser.updateView(nv);
  case 'on'
    set(APPLICATION.graphview_handle,'visible','off');
end

% --- Executes during object creation, after setting all properties.
function graph_CreateFcn(hObject, eventdata, handles)
% hObject    handle to graph (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
try
  image_pic = im2double(imread('../H36MGUI/icons/invest-applet.png'));
  set(hObject,'cdata',image_pic);%repmat(image_pic,[1, 1, 3])
  set(hObject,'String','');
catch e
end


% --- Executes on button press in depth.
function depth_Callback(hObject, eventdata, handles)
% hObject    handle to depth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global APPLICATION;
v = get(APPLICATION.depth_handle,'visible');
switch v
  case 'off'
    
		[APPLICATION.visualiser exist] = APPLICATION.visualiser.loadTOF();
		if exist
			set(APPLICATION.depth_handle,'visible','on');
			APPLICATION.visualiser.updateDepth();
		else
			ed = errordlg('Please download the depth files if available!','Error');
			set(ed, 'WindowStyle', 'modal');
			uiwait(ed);
		end
  case 'on'
    set(APPLICATION.depth_handle,'visible','off');
end

