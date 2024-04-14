// Copyright (C) 2012  Marc Vivet - marc.vivet@gmail.com
// All rights reserved.
//
//   $Revision: 9 $
//   $Date: 2012-04-20 11:31:29 +0200 (Fri, 20 Apr 2012) $
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are 
// met: 
//
// 1. Redistributions of source code must retain the above copyright notice, 
//    this list of conditions and the following disclaimer. 
// 2. Redistributions in binary form must reproduce the above copyright 
//    notice, this list of conditions and the following disclaimer in the 
//    documentation and/or other materials provided with the distribution. 
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED 
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER 
// OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// The views and conclusions contained in the software and documentation are
// those of the authors and should not be interpreted as representing 
// official policies, either expressed or implied, of the FreeBSD Project.

#include "mex.h"
#include "mexVideoReader.h"
#include "VideoReader.h"
#include <string.h>
#include <time.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}


#define _MEX_VR_MAX_MEX_VR_ 128

static CVideoReader             *g_oVR[_MEX_VR_MAX_MEX_VR_];
static bool                     g_bVerbose[_MEX_VR_MAX_MEX_VR_];
static bool                     g_bShowTime[_MEX_VR_MAX_MEX_VR_];
static int                      g_iHeight[_MEX_VR_MAX_MEX_VR_];
static int                      g_iWidth[_MEX_VR_MAX_MEX_VR_]; 
static time_t                   g_tStart = 0;
static int                      g_iNumInstances = 0;
static int                      g_iActInstance = 0;

static void ClearVideoReader ( void ) {
    for (int i = 0; i < g_iNumInstances; i ++) {
        g_iActInstance = i;
        MEX_VR_VERBOSE_INFO("Deleting Video Reader Object");
    
        if (g_oVR[i])
            delete g_oVR[i];
    
        g_oVR[i] = NULL;
	
        MEX_VR_VERBOSE_OK;
    }
}

void HelpVideoReader() {
#ifdef DEBUG	
	mexPrintf("+---------------+\n");
    mexPrintf("| DEBUG LIBRARY |\n");
	mexPrintf("+---------------+\n\n");
#endif
	
	mexPrintf("\n");
	mexPrintf(" mexVideoReader MexFunction writed in Cpp to read videos under Mac OsX using the library ffmpeg.\n");
	mexPrintf("      Description:\n");
	mexPrintf("      =============\n");
	mexPrintf("        Author: Marc Vivet - marc@vivet.cat\n");
	mexPrintf("        $Date: 2012-04-20 11:31:29 +0200 (Fri, 20 Apr 2012) $\n");
	mexPrintf("        $Revision: 9 $\n");
	mexPrintf("\n");
	mexPrintf("      Syntax:\n");
	mexPrintf("      ============\n");
	mexPrintf("\n");
	mexPrintf("        %% Create a new Video Reader Object.\n");
	mexPrintf("        info = mexVideoReader( 0, videoName, 'PorpertyName', PopertyValue, ... );\n");
	mexPrintf("\n");        
	mexPrintf("        %% Obtain a frame\n");
	mexPrintf("        %% Frame is of type double with values from 0 to 1.\n");
	mexPrintf("        frame = mexVideoReader(3, info.Id);\n"); 
	mexPrintf("\n");  
	mexPrintf("        %% Read next frame\n");
	mexPrintf("        readed = mexVideoReader(1, info.Id);\n"); 
	mexPrintf("\n");
	mexPrintf("        %% Seek to specific frame\n");
	mexPrintf("        readed = mexVideoReader(4, info.Id, frameNum);\n"); 
	mexPrintf("\n");
	mexPrintf("        %% Clear video reader object\n"); 
	mexPrintf("        mexVideoReader( 2, info.Id );\n");
	mexPrintf("        %% Or\n");
	mexPrintf("        clear all;\n");
	mexPrintf("\n");
	mexPrintf("      Configurable Properties:\n");
	mexPrintf("      =========================\n");
	mexPrintf("        +--------------------+------------------------------------------+\n");
	mexPrintf("        | Property Name      | Description                              |\n");
	mexPrintf("        +====================+==========================================+\n");
	mexPrintf("        | Vervose            | Shows the internal state of the object   |\n");
	mexPrintf("        |                    | by generating messages. (boolean)        |\n");
	mexPrintf("        +--------------------+------------------------------------------+\n");
	mexPrintf("        | ShowTime           | Shows the process time ( Verbose is      |\n");
	mexPrintf("        |                    | needed. (boolean)                        |\n");
	mexPrintf("        +--------------------+------------------------------------------+\n");
	mexPrintf("        | Scale              | Specify the size of the returned frame.  |\n");
	mexPrintf("        |                    | Its format is [width height].            |\n");
	mexPrintf("        +--------------------+------------------------------------------+\n");
	mexPrintf("\n");
    mexPrintf("     Example:\n");
	mexPrintf("     =========\n");
	mexPrintf("        clear all;\n");
	mexPrintf("\n");														
	mexPrintf("        info = mexVideoReader(0, 'videoFile.mpg', ...\n");
	mexPrintf("                                   'Scale', [720 512], ...\n");
	mexPrintf("                                   'Verbose', true, ...\n");
	mexPrintf("                                   'ShowTime', true);\n");
	mexPrintf("\n");														
	mexPrintf("        frame = mexVideoReader(3, info.Id);\n");
	mexPrintf("        figure, hi = imshow(frame); drawnow;\n");
	mexPrintf("\n");														
	mexPrintf("        while (true)\n");
	mexPrintf("           if (~mexVideoReader(1, info.Id));\n");	
	mexPrintf("               break;\n");
	mexPrintf("           end\n");	
	mexPrintf("\n");	
	mexPrintf("           frame = mexVideoReader(3, info.Id);\n");
	mexPrintf("           set(hi, 'cdata', frame); drawnow;\n");
	mexPrintf("        end\n");
	mexPrintf("\n");														
	mexPrintf("        mexVideoReader(2, info.Id);\n");
	mexPrintf("        disp(info);\n");
}

void ErrorVideoReader(int errorCode) {
    MEX_VR_VERBOSE_FAIL;
	
	char buf[512];
	
    switch (errorCode) {
        case _MEX_VRE_INCORRECT_PARAM_FORMAT:
            mexErrMsgTxt("Incorrect Video Reader parameter format. The first parameter must be int32");
            break;
        case _MEX_VRE_UKNOWN_OPTION:
            mexErrMsgTxt("Uknown Video Reader Option.");
            break;
        case _MEX_VRE_OPTION_FORMAT_INCORRECT:
            mexErrMsgTxt("Incorrect option parameter format.");
            break;
        case _MEX_VRE_OPTION_UKNOWN:
            mexErrMsgTxt("Uknown option paramater for mexVideoReader.");
            break;
        case _MEX_VRE_OPTION_VERBOSE_INCORRECT:
            mexErrMsgTxt("Verbose option must be a boolean (true, false).\n");
            break;
        case _MEX_VRE_CREATE_INCORRECT_PARAMETER:
            mexErrMsgTxt("Incorrect parameter on mexVideoReader CREATE.\n");
            break;
		case _MEX_VRE_OPTION_SHOW_TIME_INCORRECT:
            mexErrMsgTxt("ShowTime option must be a boolean (true, false)\n");
            break;
        case _MEX_VRE_INCORRECT_ID_FORMAT:
            mexErrMsgTxt("Second parameter must be the Video Reader Id (double)\n");
            break;
        case _MEX_VRE_INVALID_MEX_VR_ID:
            mexErrMsgTxt("Invalid Video Reader Id\n");
            break;
        case _MEX_VRE_INVALID_VR:
            mexErrMsgTxt("Invalid Video Reader Object\n");
            break;
		case _MEX_VRE_GO_TO_FRAME_INCORRECT_FORMAT:
            mexErrMsgTxt("Frame number must be a scalar\n");
            break;
		case _MEX_VRE_GO_TO_FRAME_NEGATIVE_FRAME:
            mexErrMsgTxt("Frame number must be positive\n");
            break;	
		case _MEX_VRE_OPTION_SIZE_INCORRECT_DIMENTIONS:
            mexErrMsgTxt("Incorrect Size parameter it must be formated as [width, height]\n");
            break;
		case _MEX_VRE_OPTION_SIZE_INCORRECT_FORMAT:
            mexErrMsgTxt("Incorrect Size parameter format, it must be [width, height]\n");
            break;    
		case _MEX_VRE_OPTION_SIZE_INVALIT_VALUE:
            mexErrMsgTxt("The minimum values for Size parameter are [10, 10]\n");
            break;       
       /* case _MEX_VRE_CREATE_CVIDEOREADER_:
            mexErrMsgTxt("\n");
            break;*/
        default:
            sprintf( buf, "%s\n", CVideoReader::TranslateError(MEX_VW_CONVERT_TO_VW_ERROR(errorCode))); 
            mexErrMsgTxt(buf);
            break;
    };
}

void CheckOptionsVideoReader ( int nrhs, const mxArray*prhs[] ) {
	char *option = NULL;
    
    double* pArray = NULL;
    const mwSize *sizeMatrix; 
	
	g_bVerbose  [g_iActInstance] = false;
	g_bShowTime [g_iActInstance] = false;
    g_iHeight   [g_iActInstance] = -1;
    g_iWidth    [g_iActInstance] = -1;
	
	for ( int i = 2; i < nrhs; i += 2 ) {
		if( mxIsChar( prhs[i] ) ) {
            option = mxArrayToString(prhs[i]); 
            
            if ( strcmp( option, "Verbose" ) == 0 ) {
                if( !mxIsLogical(prhs[i + 1]) ) ErrorVideoReader(_MEX_VRE_OPTION_VERBOSE_INCORRECT);
                else g_bVerbose[g_iActInstance] = (bool)(*(mxGetLogicals(prhs[i + 1])));
            } else if ( strcmp( option, "ShowTime" ) == 0 ) {
				if( !mxIsLogical(prhs[i + 1]) ) ErrorVideoReader(_MEX_VRE_OPTION_SHOW_TIME_INCORRECT);
                else g_bShowTime[g_iActInstance] = (bool)(*(mxGetLogicals(prhs[i + 1])));
			} else if ( strcmp( option, "Scale" ) == 0 ) {
                if ( mxGetNumberOfDimensions(prhs[i + 1]) != 2 )
                    ErrorVideoReader(_MEX_VRE_OPTION_SIZE_INCORRECT_DIMENTIONS);
                
                sizeMatrix = mxGetDimensions(prhs[i + 1]);
                
                if ( sizeMatrix[0] != 1 || sizeMatrix[1] != 2 )
                    ErrorVideoReader(_MEX_VRE_OPTION_SIZE_INCORRECT_FORMAT);
                
                pArray = (double *)mxGetPr((mxArray *)prhs[i + 1]);
                
                g_iWidth[g_iActInstance]  = (int) *(pArray);
                g_iHeight[g_iActInstance] = (int) *(pArray + 1);
                
                if ( g_iWidth[g_iActInstance] < 10 || g_iHeight[g_iActInstance] < 10 )
                    ErrorVideoReader(_MEX_VRE_OPTION_SIZE_INVALIT_VALUE);            
            } else ErrorVideoReader(_MEX_VRE_OPTION_UKNOWN);
            
            mxFree(option);
        } else ErrorVideoReader(_MEX_VRE_OPTION_FORMAT_INCORRECT);
	}
}

MEX_VR_FUNC(Create) {
    char *videoName = NULL;
    MEX_VR_RESULT vrr = _MEX_VR_NO_ERROR;
    
    g_iActInstance = g_iNumInstances;
    
    mexAtExit(ClearVideoReader);
    
    if( !mxIsChar( prhs[1] ) ) {
        return _MEX_VRE_CREATE_INCORRECT_PARAMETER;
    }
    
    CheckOptionsVideoReader ( nrhs, prhs );
	
	MEX_VR_VERBOSE_INFO("Creating new object");
    
    videoName = mxArrayToString(prhs[1]);
    
    if ( g_iWidth[g_iActInstance] != -1 ) {
        g_oVR[g_iActInstance] = new CVideoReader( videoName, g_iWidth[g_iActInstance], g_iHeight[g_iActInstance], &vrr );
    } else {
        g_oVR[g_iActInstance] = new CVideoReader(videoName, &vrr);
        g_iWidth[g_iActInstance]  = g_oVR[g_iActInstance]->GetWidth();
        g_iHeight[g_iActInstance] = g_oVR[g_iActInstance]->GetHeight();
    }
    mxFree(videoName);
    
    if ( vrr < 0 ) {
		if ( g_oVR[g_iActInstance] )
			delete g_oVR[g_iActInstance];
		
		g_oVR[g_iActInstance] = NULL;
		
		plhs[0] = mxCreateDoubleScalar(0);
		
		return MEX_VW_CONVERT_TO_MEX_ERROR(vrr);
	}
	
	const char *fieldnames[] = {"Id", "FrameWidth", "FrameHeight", "Width", "Height", "NumFrames", "Verbose", "ShowTime"};
	int nfields = 8;
	
	plhs[0] = mxCreateStructMatrix(1, 1, nfields, fieldnames);
    mxSetFieldByNumber(plhs[0], 0, 0, mxCreateDoubleScalar(g_iNumInstances));
	mxSetFieldByNumber(plhs[0], 0, 1, mxCreateDoubleScalar(g_oVR[g_iActInstance]->GetFrameWidth()));
	mxSetFieldByNumber(plhs[0], 0, 2, mxCreateDoubleScalar(g_oVR[g_iActInstance]->GetFrameHeight()));
    mxSetFieldByNumber(plhs[0], 0, 3, mxCreateDoubleScalar(g_iWidth[g_iActInstance]));
	mxSetFieldByNumber(plhs[0], 0, 4, mxCreateDoubleScalar(g_iHeight[g_iActInstance]));
	mxSetFieldByNumber(plhs[0], 0, 5, mxCreateDoubleScalar(g_oVR[g_iActInstance]->GetNumFrames()));
	mxSetFieldByNumber(plhs[0], 0, 6, mxCreateDoubleScalar(g_bVerbose[g_iActInstance]));
	mxSetFieldByNumber(plhs[0], 0, 7, mxCreateDoubleScalar(g_bShowTime[g_iActInstance]));
    
    ++g_iNumInstances;
	    
    return _MEX_VR_NO_ERROR;
}

MEX_VR_FUNC(NextFrame) {
    MEX_VR_VERBOSE_INFO("Reading next frame");
    MEX_VR_CHECK_ID;
    
    bool res = g_oVR[g_iActInstance]->NextFrame();
    plhs[0] = mxCreateLogicalScalar(res);    
    
    return _MEX_VR_NO_ERROR;
}

MEX_VR_FUNC(Delete) {
    MEX_VR_VERBOSE_INFO("Deleting Video Reader object");
    MEX_VR_CHECK_ID;   
    
    if ( g_oVR[g_iActInstance] )
        delete g_oVR[g_iActInstance];
    
    g_oVR[g_iActInstance] = NULL;
    
    return _MEX_VR_NO_ERROR;
}

MEX_VR_FUNC(GetFrame) {
    MEX_VR_VERBOSE_INFO("Obtaining current frame");
    MEX_VR_CHECK_ID;  
    
    int width = g_oVR[g_iActInstance]->GetWidth();
    int height = g_oVR[g_iActInstance]->GetHeight();
    int size = width * height;
    int deep = 3;
    
	int numDimensionsParametre = 3;
	const mwSize dimsParametre[3] = {height, width, deep};
    
	mxArray *retMatrix = NULL;
    AVPicture* pFrameRGB = NULL;
    
    g_oVR[g_iActInstance]->GetFrame ( &pFrameRGB );
    
	retMatrix = mxCreateNumericArray(numDimensionsParametre, dimsParametre, mxDOUBLE_CLASS, mxREAL);
    
	double* matOutR = (double*) mxGetPr (retMatrix);
    double* matOutG = matOutR + size;
    double* matOutB = matOutG + size;
	
	double norm = 1 / 255.0;
    
    unsigned char *pData = (unsigned char*)*(pFrameRGB->data);
	unsigned char *ppData;
	int step = (width - 1) * 3;
    
	//pData += (size - width) * 3; 
	for (int j = 0; j < width; ++j) {
		ppData = pData + 3 * j;
		for (int i = 0; i < height; ++i, ++matOutR, ++matOutG, ++matOutB, ppData += step ) {
			*matOutR = norm * (double)*ppData;
			ppData++;
			*matOutG = norm * (double)*ppData;
			ppData++;
			*matOutB = norm * (double)*ppData;              
			ppData++;    
		}
	}
	    
	plhs[0] = retMatrix;
    
    return _MEX_VR_NO_ERROR;    
}

MEX_VR_FUNC(GoToFrame) {
	MEX_VR_VERBOSE_INFO("Reading Specific Frame Number");
    MEX_VR_CHECK_ID; 
	
	if( !mxIsDouble( prhs[2] ) )
        return _MEX_VRE_GO_TO_FRAME_INCORRECT_FORMAT;
	
	double frameNum = mxGetScalar(prhs[2]);
	
	if ( frameNum < 0 )
		return _MEX_VRE_GO_TO_FRAME_NEGATIVE_FRAME;
	
	bool res = g_oVR[g_iActInstance]->GoToFrame((int) frameNum);
    plhs[0] = mxCreateLogicalScalar(res);    
	
	return _MEX_VR_NO_ERROR; 
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] ) { 
    if ( nrhs == 0 ) {
        HelpVideoReader();
        return;
    }
    
    if( !mxIsDouble( prhs[0] ) ) {
        ErrorVideoReader(_MEX_VRE_INCORRECT_PARAM_FORMAT);
    }
       
    double option = mxGetScalar(prhs[0]);	
       
    switch ((int)option) {
        case _MEX_VRF_CREATE:
            MEX_VR_CALL_FUNC(Create);
            break;
        case _MEX_VRF_NEXT_FRAME:
            MEX_VR_CALL_FUNC(NextFrame);
            break;
        case _MEX_VRF_DELETE:
            MEX_VR_CALL_FUNC(Delete);
            break;
        case _MEX_VRF_GET_FRAME:
            MEX_VR_CALL_FUNC(GetFrame);
            break;
		case _MEX_VRF_GO_TO_FRAME:
			MEX_VR_CALL_FUNC(GoToFrame);
			break;
        default:
            ErrorVideoReader(_MEX_VRE_UKNOWN_OPTION);
            break;               
    };
	
	MEX_VR_VERBOSE_OK;
}
