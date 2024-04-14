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


#ifndef __VIDEOWRITER__
#define __VIDEOWRITER__ 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "VideoWriterDefinitions.h"

#ifdef _WIN32
#define snprintf sprintf_s 

#ifndef INT64_C
#define INT64_C(c) (c ## LL)
#define UINT64_C(c) (c ## ULL)
#endif
#else
#ifdef __cplusplus
#define __STDC_CONSTANT_MACROS
#ifdef _STDINT_H
#undef _STDINT_H
#endif
# include <stdint.h>
#endif
#endif

extern "C" {
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
}

#define VW_RESULT						int

class CVideoWriter {
	char* m_sFileName;
    AVOutputFormat* m_pFmt;
    AVFormatContext* m_pOc;		
	
	int m_iWidth;
	int m_iHeight;
	int m_iFps;
	int m_iBitRate;
	
	/**************************************************************/
	/* video output */
	
	AVFrame* m_pPicture;
	uint8_t* m_ucVideo_outbuf;
	int m_iFrame_count;
	int video_outbuf_size;
	
	AVStream* m_pVideo_st;
	double m_dVideo_pts;
	
	VW_RESULT AddVideoStream( void );
	VW_RESULT OpenVideo( void );
	VW_RESULT WriteVideoFrame( void );
	VW_RESULT CloseVideo( void );
    
public:
        
    CVideoWriter( const char* p_sFileName, int p_iWidth, int p_iHeight, int p_iFps, int p_iBitRate, char* p_pFormat, VW_RESULT* p_iResult );
	VW_RESULT AddFrameRGB24 ( AVPicture* p_pPicture, int p_iWidth, int p_iHeight );
    ~CVideoWriter( void );
	
	static const char* TranslateError ( VW_RESULT error );
};

#endif