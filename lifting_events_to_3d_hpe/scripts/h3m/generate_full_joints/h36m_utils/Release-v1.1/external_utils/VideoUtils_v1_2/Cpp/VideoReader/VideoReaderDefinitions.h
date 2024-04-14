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


#ifndef _VIDEO_READER_DEFINITIONS_
#define _VIDEO_READER_DEFINITIONS_

#define MEX_VR_RESULT                                           int

#define _MEX_VRF_CREATE                                         0
#define _MEX_VRF_NEXT_FRAME                                     1
#define _MEX_VRF_DELETE                                         2
#define _MEX_VRF_GET_FRAME                                      3
#define _MEX_VRF_GO_TO_FRAME									4

#define _MEX_VR_NO_ERROR                                        0

#define _MEX_VRE_INCORRECT_PARAM_FORMAT                         -1
#define _MEX_VRE_UKNOWN_OPTION                                  -2
#define _MEX_VRE_OPTION_FORMAT_INCORRECT                        -3
#define _MEX_VRE_OPTION_UKNOWN                                  -4
#define _MEX_VRE_OPTION_VERBOSE_INCORRECT                       -5
#define _MEX_VRE_CREATE_INCORRECT_PARAMETER                     -6
#define _MEX_VRE_OPTION_SHOW_TIME_INCORRECT						-7
#define _MEX_VRE_INCORRECT_ID_FORMAT                            -8
#define _MEX_VRE_INVALID_MEX_VR_ID                              -9
#define _MEX_VRE_GO_TO_FRAME_INCORRECT_FORMAT					-10
#define _MEX_VRE_GO_TO_FRAME_NEGATIVE_FRAME						-11
#define _MEX_VRE_INVALID_VR										-12
#define _MEX_VRE_OPTION_SIZE_INCORRECT_DIMENTIONS               -13
#define _MEX_VRE_OPTION_SIZE_INCORRECT_FORMAT                   -14
#define _MEX_VRE_OPTION_SIZE_INVALIT_VALUE                      -15

#define MEX_VW_CONVERT_TO_MEX_ERROR(x)							(x - 100)
#define MEX_VW_CONVERT_TO_VW_ERROR(x)							(x + 100)

#define MEX_VR_FUNC(f)                          MEX_VR_RESULT MEX_VR_FUNC_##f( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] )
#define MEX_VR_CALL_FUNC(f)                     {MEX_VR_RESULT result = MEX_VR_FUNC_##f(nlhs, plhs, nrhs, prhs); \
												 if ( result < 0 ) ErrorVideoReader(result);}
#define MEX_VR_VERBOSE_INFO(inf)				{if ( g_bVerbose[g_iActInstance] ) { \
												 mexPrintf("[mexVideoReader ~ %d] >> %s ... ", g_iActInstance, inf); \
												 if ( g_bShowTime[g_iActInstance] ) g_tStart = clock();\
												}}
#define MEX_VR_VERBOSE_OK						{if ( g_bVerbose[g_iActInstance] ) { \
												 if ( g_bShowTime[g_iActInstance] ) { \
													mexPrintf("OK! - %fs\n", ((double) (clock() - g_tStart) / (double)CLOCKS_PER_SEC)); \
												 } else { \
													mexPrintf("OK!\n"); \
												 } \
												}}
#define MEX_VR_VERBOSE_FAIL						{if ( g_bVerbose[g_iActInstance] ) { \
												 if ( g_bShowTime[g_iActInstance] ) { \
													mexPrintf("FAIL! - %fs\n", ((double) (clock() - g_tStart) / (double)CLOCKS_PER_SEC)); \
												 } else { \
													mexPrintf("FAIL!\n"); \
												 } \
												}}

#define MEX_VR_CHECK_ID                         {if( !mxIsDouble( prhs[1] ) ) \
												 return _MEX_VRE_INCORRECT_ID_FORMAT; \
												 double id = mxGetScalar(prhs[1]); \
												 g_iActInstance = id; \
												 if ( g_iActInstance >= g_iNumInstances ) \
													return _MEX_VRE_INVALID_MEX_VR_ID; \
												 if ( g_oVR[g_iActInstance] == NULL ) \
													return _MEX_VRE_INVALID_VR;}

#endif
