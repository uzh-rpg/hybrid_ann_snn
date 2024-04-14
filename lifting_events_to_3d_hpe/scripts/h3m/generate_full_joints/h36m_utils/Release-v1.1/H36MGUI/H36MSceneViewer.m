function varargout = H36MSceneViewer(varargin)
% H36MSCENEVIEWER MATLAB code for H36MSceneViewer.fig
%      H36MSCENEVIEWER, by itself, creates a new H36MSCENEVIEWER or raises the existing
%      singleton*.
%
%      H = H36MSCENEVIEWER returns the handle to a new H36MSCENEVIEWER or the handle to
%      the existing singleton*.
%
%      H36MSCENEVIEWER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in H36MSCENEVIEWER.M with the given input arguments.
%
%      H36MSCENEVIEWER('Property','Value',...) creates a new H36MSCENEVIEWER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before H36MSceneViewer_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to H36MSceneViewer_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help H36MSceneViewer

% Last Modified by GUIDE v2.5 24-Oct-2012 15:19:28

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @H36MSceneViewer_OpeningFcn, ...
                   'gui_OutputFcn',  @H36MSceneViewer_OutputFcn, ...
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


% --- Executes just before H36MSceneViewer is made visible.
function H36MSceneViewer_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to H36MSceneViewer (see VARARGIN)

% Choose default command line output for H36MSceneViewer
handles.output = hObject;

% Update handles structure

guidata(hObject, handles);
global APPLICATION;
APPLICATION.visualiser=APPLICATION.visualiser.setSceneView(handles);
set(hObject,'Renderer','OpenGL')


% set(hObject,'Renderer','zbuffer')
% UIWAIT makes H36MSceneViewer wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = H36MSceneViewer_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes during object deletion, before destroying properties.
function figure1_DeleteFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% set(hObject,'visible','off');


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
set(hObject,'visible','off');
