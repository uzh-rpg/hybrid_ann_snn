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


#ifndef __VIDEOREADER__
#define __VIDEOREADER__ 

#ifdef __cplusplus
#define __STDC_CONSTANT_MACROS
#ifdef _STDINT_H
#undef _STDINT_H
#endif
# include <stdint.h>
#endif

#include <stdlib.h>

#define VR_RESULT                            int

//#define int64_t long long

extern "C" {
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
}
/*

typedef struct AVFormatContext;
typedef struct AVCodecContext;
typedef struct AVFrame;
typedef struct AVPicture;
*/
class CVideoReader {
    AVFormatContext *m_pFormatCtx;
    int             m_iVideoStream;
    AVCodecContext  *m_pCodecCtx;
    AVFrame         *m_pFrame;    
    AVPicture       *m_pFrameRGB;
	
	bool m_bPictureAllocated;
    
    struct SwsContext *m_pImg_convert_ctx;
    
    char* m_sFileName;
    int m_iWidth;
    int m_iHeight;
    
    int m_iFrameWidth;
    int m_iFrameHeight;
    
    int m_iFrameNumber;
    int m_iTotalNumFrames;
    
    int64_t m_iFrameTime;
    int64_t m_iInterFrameTime;
    double m_iSecond;
    double m_dFrameTime2Sec;
    
    bool m_bFrameConverted;
    
    int InitFfmpeg ( void );
    int InitFrameSize( int p_iWidth, int p_iHeight );
    int ConvertFrameToRGB( void );
    
public:
	static const char* TranslateError ( VR_RESULT error );
	
    CVideoReader( const char* p_sFileName );
    CVideoReader( const char* p_sFileName, VR_RESULT* p_iResult );
	CVideoReader( const char* p_sFileName, int p_iWidth, int p_iHeight  );
    CVideoReader( const char* p_sFileName, int p_iWidth, int p_iHeight, VR_RESULT* p_iResult  );
    ~CVideoReader( void );
    
    VR_RESULT Restard ( void );
    
    bool NextFrame ( void );
    
    VR_RESULT GetFrame ( AVPicture** p_pFrameRGB );
    VR_RESULT GetFrame ( unsigned char** p_ucData, int* p_iWidth, int* p_iHeight, int* p_iLineSize );
	
    bool GoToFrame ( int p_iFrameNume );
    bool GoToSecond ( double p_dSecond );
    bool GoToTime ( int64_t p_Time );
    
    int GetWidth();
    int GetHeight();
    int GetFrameWidth();
    int GetFrameHeight();
    int64_t GetTime ( void );
    int GetFrameNumber();
    int GetNumFrames();
    int GetSeconds();
};

#endif

