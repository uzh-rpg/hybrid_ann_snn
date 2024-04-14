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
#include "mexVideoWriter.h"
#include "VideoWriter.h"
#include <string.h>
#include <time.h>

#define _MEX_VW_MAX_VW_ 128

static CVideoWriter             *g_oVR		[_MEX_VW_MAX_VW_];
static bool                     g_bVerbose	[_MEX_VW_MAX_VW_];
static bool                     g_bShowTime	[_MEX_VW_MAX_VW_];
static int                      g_iHeight	[_MEX_VW_MAX_VW_];
static int                      g_iWidth	[_MEX_VW_MAX_VW_]; 
static int                      g_iFps		[_MEX_VW_MAX_VW_]; 
static int                      g_iBps		[_MEX_VW_MAX_VW_];
static int                      g_iFormat	[_MEX_VW_MAX_VW_];

static time_t                   g_tStart = 0;
static int                      g_iNumInstances = 0;
static int                      g_iActInstance = 0;

static void ClearVideoWriter ( void ) {
    for (int i = 0; i < g_iNumInstances; i ++) {
        g_iActInstance = i;
        MEX_VW_VERBOSE_INFO("Deleting Video Writer Object");
    
        if (g_oVR[i])
            delete g_oVR[i];
    
        g_oVR[i] = NULL;
	
        MEX_VW_VERBOSE_OK;
    }
}

void HelpVideoWriter() {
#ifdef DEBUG	
	mexPrintf("+---------------+\n");
    mexPrintf("| DEBUG LIBRARY |\n");
	mexPrintf("+---------------+\n\n");
#endif
	
	mexPrintf("\n");
	mexPrintf(" mexVideoWriter MexFunction writed in Cpp to create videos under Mac OsX using the library ffmpeg.\n");
	mexPrintf("      Description:\n");
	mexPrintf("      =============\n");
	mexPrintf("        Author: Marc Vivet - marc@vivet.cat\n");
	mexPrintf("        $Date: 2012-04-20 11:31:29 +0200 (Fri, 20 Apr 2012) $\n");
	mexPrintf("        $Revision: 9 $\n");
	mexPrintf("\n");
	mexPrintf("      Syntax:\n");
	mexPrintf("      ============\n");
	mexPrintf("\n");
	mexPrintf("        %% Create a new Video Writer Object.\n");
	mexPrintf("        info = mexVideoWriter( 0, videoName, 'PorpertyName', PopertyValue, ... );\n");
	mexPrintf("\n");        
	mexPrintf("        %% Adding a Frame\n");
	mexPrintf("        %% Frame must be double with values from 0 to 1.\n");
	mexPrintf("        mexVideoWriter( 1, info.Id, frame );\n"); 
	mexPrintf("\n");        
	mexPrintf("        %% End Video\n"); 
	mexPrintf("        mexVideoWriter( 2, info.Id );\n");
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
	mexPrintf("        | Size               | Specify the resultant video size. The    |\n");
	mexPrintf("        |                    | input format is [width height].          |\n");
	mexPrintf("        |                    | By default is 720x512.                   |\n");
	mexPrintf("        +--------------------+------------------------------------------+\n");
	mexPrintf("        | Fps                | Determines the frame rate.               |\n");
	mexPrintf("        |                    | By default is 25 frames per second.      |\n");
	mexPrintf("        +--------------------+------------------------------------------+\n");
	mexPrintf("        | Bps                | Determines the quality of the resultant  |\n");
	mexPrintf("        |                    | video. By default is 720x512x3 (Hight    |\n");
	mexPrintf("        |                    | quality).                                |\n");
	mexPrintf("        +--------------------+------------------------------------------+\n");
	mexPrintf("        | Format             | Determines the video encoder.            |\n");
	mexPrintf("        |                    | The suported encoders are:               |\n");
	mexPrintf("        |                    |                mpg - MPEG 1              |\n");
	mexPrintf("        |                    |                dvd - MPEG 2              |\n");		
    mexPrintf("        |                    |                mp4 - MPEG 4              |\n");
	mexPrintf("        |                    |                mov - Quick Time          |\n");			
	mexPrintf("        |                    |                wmv - Windows media video |\n");
	mexPrintf("        |                    |                flv - Flash Video         |\n");			
	mexPrintf("        +--------------------+------------------------------------------+\n");
	mexPrintf("\n");
    mexPrintf("     Example:\n");
	mexPrintf("     =========\n");
	mexPrintf("        clear all;\n");
	mexPrintf("\n");														
	mexPrintf("        info = mexVideoWriter  (0, 'Test', ...\n");
	mexPrintf("                                   'Format', 'mov', ...\n");
	mexPrintf("                                   'Size', [720 512], ...\n");
	mexPrintf("                                   'Fps', 25, ...\n");
	mexPrintf("                                   'Bps', 400000, ...\n");
	mexPrintf("                                   'Verbose', true, ...\n");
	mexPrintf("                                   'ShowTime', true);\n");
	mexPrintf("\n");														
	mexPrintf("        img = double(imresize(imread('peppers.png'), [480 640])) / 255.0;\n");
	mexPrintf("\n");														
	mexPrintf("        for i = 1:100\n");
	mexPrintf("           img = double(imresize(imread('peppers.png'), [4 * i 10 * i])) / 255.0;\n");
	mexPrintf("           mexVideoWriter(1, 0, img);\n");
	mexPrintf("        end\n");
	mexPrintf("\n");														
	mexPrintf("        mexVideoWriter(2, 0);\n");
	mexPrintf("        disp(info);\n");
}

void ErrorVideoWriter(int errorCode) {
    MEX_VW_VERBOSE_FAIL;
	
	char buf[512];
	
    switch (errorCode) {
        case _MEX_VWE_INCORRECT_PARAM_FORMAT:
            mexErrMsgTxt("Incorrect Video Writer parameter format\n");
            break;
		case _MEX_VWE_UKNOWN_FUNCTION:
			mexErrMsgTxt("Function Uknown\n");
			break;
        case _MEX_VWE_OPTION_FORMAT_INCORRECT:
            mexErrMsgTxt("Incorrect option parameter format\n");
            break;
        case _MEX_VWE_OPTION_UKNOWN:
            mexErrMsgTxt("Uknown option paramater for mexVideoWriter\n");
            break;
        case _MEX_VWE_OPTION_VERBOSE_INCORRECT:
            mexErrMsgTxt("Verbose option must be a boolean (true, false)\n");
            break;
        case _MEX_VWE_CREATE_INCORRECT_PARAMETER:
            mexErrMsgTxt("Incorrect parameter on mexVideoWriter CREATE\n");
            break;
		case _MEX_VWE_OPTION_SHOW_TIME_INCORRECT:
            mexErrMsgTxt("ShowTime option must be a boolean (true, false)\n");
            break;
        case _MEX_VWE_INCORRECT_ID_FORMAT:
            mexErrMsgTxt("Second parameter must be the Video Reader Id (double)\n");
            break;
        case _MEX_VWE_INVALID_VW_ID:
            mexErrMsgTxt("Invalid Video Writer Id\n");
            break;
		case _MEX_VWE_OPTION_SIZE_INCORRECT_DIMENTIONS:
            mexErrMsgTxt("Incorrect Size parameter it must be formated as [width, height], where width and height are multiple of 2\n");
            break;
		case _MEX_VWE_OPTION_SIZE_INCORRECT_FORMAT:
            mexErrMsgTxt("Incorrect Size parameter format, it must be [width, height]\n");
            break;    
		case _MEX_VWE_OPTION_SIZE_INVALIT_VALUE:
            mexErrMsgTxt("The minimum values for Size parameter are [10, 10]\n");
            break; 			
		case _MEX_VWE_OPTION_FPS_INVALIT_VALUE:
            mexErrMsgTxt("Frames per second parameter must be positive and greater than 0\n");
            break; 
		case _MEX_VWE_OPTION_BPS_INVALIT_VALUE:
            mexErrMsgTxt("Bits per second parameter must be positive and greater than 1000\n");
            break; 
		case _MEX_VWE_OPTION_FORMAT_INVALIT_VALUE:
            mexErrMsgTxt("Uknown file format, the formats suported are [mpg, dvd, mp4, mov, wmv and flv]\n");
            break; 
        default:
			sprintf( buf, "%s\n", CVideoWriter::TranslateError(MEX_VW_CONVERT_TO_VW_ERROR(errorCode))); 
            mexErrMsgTxt(buf);
            break;
    };
}

int Format2Id ( char* pFormat ) {
	char form[8];
	int i = 0;
	int c;
	
	while (pFormat[i] && i < 8)
	{
		c = (int)pFormat[i];
		form[i] = (char)(tolower(c));
		i++;
	}
	
	form[i] = '\0';
	
		 if ( strcmp( form, "mpg"	) == 0 ) return 0;
	else if ( strcmp( form, "dvd" ) == 0 ) return 1;
	else if ( strcmp( form, "mp4"	) == 0 ) return 2;
	else if ( strcmp( form, "mov"	) == 0 ) return 3;
	else if ( strcmp( form, "wmv"	) == 0 ) return 4;
	else if ( strcmp( form, "flv"	) == 0 ) return 5;
	
	return -1;
}

void Id2Format ( int id, char* pFormat ) {
	switch ( id ) {
		case 0:
			strcpy( pFormat, "mpg\0" );
			break;
		case 1:
			strcpy( pFormat, "dvd\0" );
			break;
		case 2:
			strcpy( pFormat, "mp4\0" );
			break;
		case 3:
			strcpy( pFormat, "mov\0" );
			break;
		case 4:
			strcpy( pFormat, "wmv\0" );
			break;
		case 5:
			strcpy( pFormat, "flv\0" );
			break;
	};
}

void CheckOptionsVideoWriter ( int nrhs, const mxArray*prhs[] ) {
	char *option = NULL;
    
    double* pArray = NULL;
    const mwSize *sizeMatrix; 
	char *format = NULL;
	
	g_bVerbose  [g_iActInstance] = false;
	g_bShowTime [g_iActInstance] = false;
    g_iHeight   [g_iActInstance] = 512;
    g_iWidth    [g_iActInstance] = 720;
	g_iFps		[g_iActInstance] = 25;
	g_iBps		[g_iActInstance] = 720 * 512 * 3;
	g_iFormat	[g_iActInstance] = 3;
	
	for ( int i = 2; i < nrhs; i += 2 ) {
		if( mxIsChar( prhs[i] ) ) {
            option = mxArrayToString(prhs[i]); 
            
            if ( strcmp( option, "Verbose" ) == 0 ) {
                if( !mxIsLogical(prhs[i + 1]) ) ErrorVideoWriter(_MEX_VWE_OPTION_VERBOSE_INCORRECT);
                else g_bVerbose[g_iActInstance] = (bool)(*(mxGetLogicals(prhs[i + 1])));
            } else if ( strcmp( option, "ShowTime" ) == 0 ) {
				if( !mxIsLogical(prhs[i + 1]) ) ErrorVideoWriter(_MEX_VWE_OPTION_SHOW_TIME_INCORRECT);
                else g_bShowTime[g_iActInstance] = (bool)(*(mxGetLogicals(prhs[i + 1])));
			} else if ( strcmp( option, "Size" ) == 0 ) {
                if ( mxGetNumberOfDimensions(prhs[i + 1]) != 2 )
                    ErrorVideoWriter(_MEX_VWE_OPTION_SIZE_INCORRECT_DIMENTIONS);
                
                sizeMatrix = mxGetDimensions(prhs[i + 1]);
                
                if ( sizeMatrix[0] != 1 || sizeMatrix[1] != 2 )
                    ErrorVideoWriter(_MEX_VWE_OPTION_SIZE_INCORRECT_FORMAT);
                
                pArray = (double *)mxGetPr((mxArray *)prhs[i + 1]);
                
                g_iWidth[g_iActInstance]  = (int) *(pArray);
                g_iHeight[g_iActInstance] = (int) *(pArray + 1);
                
                if ( g_iWidth[g_iActInstance] < 10 || g_iHeight[g_iActInstance] < 10 )
                    ErrorVideoWriter(_MEX_VWE_OPTION_SIZE_INVALIT_VALUE);            
            } else if (strcmp( option, "Fps" ) == 0 ) {
				g_iFps[g_iActInstance]  = (int)*((double *)mxGetPr((mxArray *)prhs[i + 1]));
				
				if ( g_iFps[g_iActInstance] <= 0 ) 
					ErrorVideoWriter(_MEX_VWE_OPTION_FPS_INVALIT_VALUE);
			} else if (strcmp( option, "Bps" ) == 0 ) {
				g_iBps[g_iActInstance]  = (int)*((double *)mxGetPr((mxArray *)prhs[i + 1]));
				
				if ( g_iBps[g_iActInstance] <= 1000 )
					ErrorVideoWriter(_MEX_VWE_OPTION_BPS_INVALIT_VALUE);
			} else if (strcmp( option, "Format" ) == 0 ) {
				format = mxArrayToString(prhs[i + 1]);
				g_iFormat[g_iActInstance] = Format2Id ( format );
				mxFree(format);
				
				if ( g_iFormat[g_iActInstance] < 0 )
					ErrorVideoWriter(_MEX_VWE_OPTION_FORMAT_INVALIT_VALUE);
				
			} else ErrorVideoWriter(_MEX_VWE_OPTION_UKNOWN);
            
            mxFree(option);
        } else ErrorVideoWriter(_MEX_VWE_OPTION_FORMAT_INCORRECT);
	}
}

MEX_VW_FUNC(Create) {
    char *videoName = NULL;
    MEX_VW_RESULT vrr = _MEX_VWR_NO_ERROR;
    
    g_iActInstance = g_iNumInstances;
    
    mexAtExit(ClearVideoWriter);
    
    if( !mxIsChar( prhs[1] ) ) {
        return _MEX_VWE_CREATE_INCORRECT_PARAMETER;
    }
    
    CheckOptionsVideoWriter ( nrhs, prhs );
	
	MEX_VW_VERBOSE_INFO("Creating new Video Writer object");
    
    videoName = mxArrayToString(prhs[1]);
	char buf[256];
	char format[8];
	char fileName[512];
	
	Id2Format ( g_iFormat[g_iActInstance], format );
	
	sprintf(fileName, "%s.%s", videoName, format );
	
    g_oVR[g_iActInstance] = new CVideoWriter(fileName, g_iWidth[g_iActInstance], g_iHeight[g_iActInstance], g_iFps[g_iActInstance], g_iBps[g_iActInstance], buf, &vrr);

    mxFree(videoName);
    
    if ( vrr < 0 ) {
		if ( g_oVR[g_iActInstance] )
			delete g_oVR[g_iActInstance];
		
		g_oVR[g_iActInstance] = NULL;
		
		plhs[0] = mxCreateScalarDouble(0);
		
		return vrr;
	}
	
	const char *fieldnames[] = {"Id", "FileName", "Width", "Height", "Fps", "Bps", "Format", "Verbose", "ShowTime"};
	int nfields = 9;
	
	plhs[0] = mxCreateStructMatrix(1, 1, nfields, fieldnames);
    mxSetFieldByNumber(plhs[0], 0, 0, mxCreateScalarDouble(g_iNumInstances));
	mxSetFieldByNumber(plhs[0], 0, 1, mxCreateString(fileName));
	mxSetFieldByNumber(plhs[0], 0, 2, mxCreateScalarDouble(g_iWidth[g_iActInstance]));
	mxSetFieldByNumber(plhs[0], 0, 3, mxCreateScalarDouble(g_iHeight[g_iActInstance]));
    mxSetFieldByNumber(plhs[0], 0, 4, mxCreateScalarDouble(g_iFps[g_iActInstance]));
	mxSetFieldByNumber(plhs[0], 0, 5, mxCreateScalarDouble(g_iBps[g_iActInstance]));
	mxSetFieldByNumber(plhs[0], 0, 6, mxCreateString(buf));
	mxSetFieldByNumber(plhs[0], 0, 7, mxCreateScalarDouble(g_bVerbose[g_iActInstance]));
	mxSetFieldByNumber(plhs[0], 0, 8, mxCreateScalarDouble(g_bShowTime[g_iActInstance]));
    
    ++g_iNumInstances;
	    
    return _MEX_VWR_NO_ERROR;
}

MEX_VW_FUNC(AddFrame) {
    MEX_VW_VERBOSE_INFO("Adding a frame");
    MEX_VW_CHECK_ID;
	
	AVPicture pFrameRGB;
	int iWidth, iHeight, iChannels;
	double* pFrame;
	
	/// CAPTURING FRAME
	
	mwSize numDims;
	const mwSize *sizeMatrix; 
	
	numDims = mxGetNumberOfDimensions(prhs[2]);
	sizeMatrix = mxGetDimensions(prhs[2]);
	
	iHeight = (int)sizeMatrix[0];
	iWidth  = (int)sizeMatrix[1];
	iChannels = 1;
	
	if (numDims == 3)
		iChannels = (int)sizeMatrix[2];
	
	pFrame = (double *)mxGetPr((mxArray *)prhs[2]);
	
	avpicture_alloc(&pFrameRGB, PIX_FMT_RGB24, iWidth, iHeight);
	
	int size = iWidth * iHeight;
	
	double* matOutR = pFrame;
    double* matOutG = matOutR + size;
    double* matOutB = matOutG + size;
	
	double norm = 255.0;
	
	unsigned char *pData = (unsigned char*)*(pFrameRGB.data);
	unsigned char *ppData;
	int step = (iWidth - 1) * 3;
    
	//pData += (size - width) * 3; 
	for (int j = 0; j < iWidth; ++j) {
		ppData = pData + 3 * j;
		for (int i = 0; i < iHeight; ++i, ++matOutR, ++matOutG, ++matOutB, ppData += step ) {
			*ppData = (unsigned char) norm * *matOutR;
			ppData++;
			*ppData = (unsigned char) norm * *matOutG;
			ppData++;
			*ppData = (unsigned char) norm * *matOutB;
			ppData++;   
		}
	}
	
	g_oVR[g_iActInstance]->AddFrameRGB24(&pFrameRGB, iWidth, iHeight);
	
	avpicture_free(&pFrameRGB);
    
    return _MEX_VWR_NO_ERROR;
}

MEX_VW_FUNC(Delete) {
    MEX_VW_VERBOSE_INFO("Deleting Video Writer object");
    MEX_VW_CHECK_ID;   
    
    if ( g_oVR[g_iActInstance] )
        delete g_oVR[g_iActInstance];
    
    g_oVR[g_iActInstance] = NULL;
    
    return _MEX_VWR_NO_ERROR;
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] ) { 
    if ( nrhs == 0 ) {
        HelpVideoWriter();
        return;
    }
    
    if( !mxIsDouble( prhs[0] ) ) {
        ErrorVideoWriter(_MEX_VWE_INCORRECT_PARAM_FORMAT);
    }
       
    double option = mxGetScalar(prhs[0]);	
       
    switch ((int)option) {
        case _MEX_VWF_CREATE:
            MEX_VW_CALL_FUNC(Create);
            break;
        case _MEX_VWF_ADD_FRAME:
            MEX_VW_CALL_FUNC(AddFrame);
            break;
        case _MEX_VWF_DELETE:
            MEX_VW_CALL_FUNC(Delete);
            break;
        default:
            ErrorVideoWriter(_MEX_VWE_UKNOWN_FUNCTION);
            break;               
    };
	
	MEX_VW_VERBOSE_OK;
}
