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


#include "VideoReader.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

#ifndef CODEC_TYPE_VIDEO
#define CODEC_TYPE_VIDEO   AVMEDIA_TYPE_VIDEO 
#endif

#ifndef FFM_PACKET_SIZE
#define FFM_PACKET_SIZE 4096
#endif


#include <string.h>
#include <time.h>

#define _VR_NO_ERROR											0

#define _VRE_CREATE_FILE_NOT_FOUND								-1
#define _VRE_CREATE_STREAM_NOT_FOUND							-2
#define _VRE_CREATE_NO_VIDEO_STREAM								-3
#define _VRE_CREATE_CODEC_NOT_SUPPORTED							-4
#define _VRE_CREATE_CODEC_NOT_OPEN								-5
#define _VRE_UKNOWN_ERROR										-6

#define _VRE_TEXT_CREATE_CVIDEOREADER_FILE_NOT_FOUND			"Couldn't open file"
#define _VRE_TEXT_CREATE_CVIDEOREADER_STREAM_NOT_FOUND			"Couldn't find stream information"
#define _VRE_TEXT_CREATE_CVIDEOREADER_NO_VIDEO_STREAM			"Didn't find a video stream"
#define _VRE_TEXT_CREATE_CVIDEOREADER_CODEC_NOT_SUPPORTED		"Unsupported codec!"
#define _VRE_TEXT_CREATE_CVIDEOREADER_CODEC_NOT_OPEN			"Could not open codec"
#define _VRE_TEXT_UKNOWN_ERROR									"Uknown error"

#define VR_CHECK(x)						{	VR_RESULT tempRes = x; \
if ( tempRes < 0 ) return tempRes; }

CVideoReader::CVideoReader( const char* p_sFileName ) {
	m_bPictureAllocated = false;
	
	m_pFormatCtx = NULL;
    m_pCodecCtx = NULL;
    m_pFrame = NULL;   
	m_pImg_convert_ctx = NULL;
	
    m_sFileName = (char*) malloc (sizeof(char) * (strlen(p_sFileName) + 1));
    strcpy(m_sFileName, p_sFileName);       
	
    VR_RESULT res = InitFfmpeg();
    if ( res == _VR_NO_ERROR ) {
        InitFrameSize( m_pCodecCtx->width, m_pCodecCtx->height );
		//NextFrame();
	}
}


CVideoReader::CVideoReader( const char* p_sFileName, VR_RESULT* p_iResult ) {
	m_bPictureAllocated = false;
	
	m_pFormatCtx = NULL;
    m_pCodecCtx = NULL;
    m_pFrame = NULL;   
	m_pImg_convert_ctx = NULL;
	
    m_sFileName = (char*) malloc (sizeof(char) * (strlen(p_sFileName) + 1));
    strcpy(m_sFileName, p_sFileName);       
	
    *p_iResult = InitFfmpeg();
    if ( *p_iResult == _VR_NO_ERROR ) {
        *p_iResult = InitFrameSize( m_pCodecCtx->width, m_pCodecCtx->height );
		//NextFrame();
	}
}

CVideoReader::CVideoReader( const char* p_sFileName, int p_iWidth, int p_iHeight, VR_RESULT* p_iResult  ) {  
	m_bPictureAllocated = false;
	
	m_pFormatCtx = NULL;
    m_pCodecCtx = NULL;
    m_pFrame = NULL;    
	m_pImg_convert_ctx = NULL;
	
    m_sFileName = (char*) malloc (sizeof(char) * (strlen(p_sFileName) + 1));
    strcpy(m_sFileName, p_sFileName);
	
    *p_iResult = InitFfmpeg();
    if ( *p_iResult == _VR_NO_ERROR ) {
        *p_iResult = InitFrameSize( p_iWidth, p_iHeight );
		//NextFrame();
	}
}

CVideoReader::CVideoReader( const char* p_sFileName, int p_iWidth, int p_iHeight  ) {  
	VR_RESULT iResult = _VR_NO_ERROR;
	
	m_bPictureAllocated = false;
	
	m_pFormatCtx = NULL;
    m_pCodecCtx = NULL;
    m_pFrame = NULL;    
	m_pImg_convert_ctx = NULL;
	
    m_sFileName = (char*) malloc (sizeof(char) * (strlen(p_sFileName) + 1));
    strcpy(m_sFileName, p_sFileName);
	
    iResult = InitFfmpeg();
    if ( iResult == _VR_NO_ERROR ) {
        InitFrameSize( p_iWidth, p_iHeight );
		//NextFrame();
	}
}

const char* CVideoReader::TranslateError ( VR_RESULT error ) {
	
	switch ((int) error ) {
		case _VRE_CREATE_FILE_NOT_FOUND:
			return _VRE_TEXT_CREATE_CVIDEOREADER_FILE_NOT_FOUND;
		case _VRE_CREATE_STREAM_NOT_FOUND:
			return _VRE_TEXT_CREATE_CVIDEOREADER_STREAM_NOT_FOUND;
		case _VRE_CREATE_NO_VIDEO_STREAM:
			return _VRE_TEXT_CREATE_CVIDEOREADER_NO_VIDEO_STREAM;
		case _VRE_CREATE_CODEC_NOT_SUPPORTED:
			return _VRE_TEXT_CREATE_CVIDEOREADER_CODEC_NOT_SUPPORTED;	
		case _VRE_CREATE_CODEC_NOT_OPEN:
			return _VRE_TEXT_CREATE_CVIDEOREADER_CODEC_NOT_OPEN;			
		case _VRE_UKNOWN_ERROR:							
		default:
			return _VRE_TEXT_UKNOWN_ERROR;
	};
}


int CVideoReader::Restard ( void ) {
    
    // Close the codec
    avcodec_close(m_pCodecCtx);
    
    // Close the video file
    av_close_input_file(m_pFormatCtx);
	
    InitFfmpeg();
	
	NextFrame();
    
    return _VR_NO_ERROR;
}

int CVideoReader::InitFfmpeg ( void ) {
    int             i;
    AVCodec         *pCodec;
    
    m_iFrameNumber = 0;
    m_iFrameTime = 0;
    
    m_bFrameConverted = false;
	
    // Register all formats and codecs
    av_register_all();
    
    // Open video file
    if(av_open_input_file(&m_pFormatCtx, m_sFileName, NULL, FFM_PACKET_SIZE, NULL)!=0) {
        return _VRE_CREATE_FILE_NOT_FOUND;        
    }
    //     return -1; // Couldn't open file
    
    // Retrieve stream information
    if(av_find_stream_info(m_pFormatCtx)<0) {
        return _VRE_CREATE_STREAM_NOT_FOUND;
    }
    
    // Find the first video stream
    m_iVideoStream=-1;
    for(i=0; i<m_pFormatCtx->nb_streams; i++)
        if(m_pFormatCtx->streams[i]->codec->codec_type==CODEC_TYPE_VIDEO) {
            m_iVideoStream=i;
            break;
        }
    if(m_iVideoStream==-1) {
        return _VRE_CREATE_NO_VIDEO_STREAM;
        //return -1; // Didn't find a video stream
    }
    
    // Get a pointer to the codec context for the video stream
    m_pCodecCtx=m_pFormatCtx->streams[m_iVideoStream]->codec;
    
    // Find the decoder for the video stream
    pCodec=avcodec_find_decoder(m_pCodecCtx->codec_id);
    if(pCodec==NULL) {
        return _VRE_CREATE_CODEC_NOT_SUPPORTED; // Codec not found
    }
    // Open codec
    if(avcodec_open(m_pCodecCtx, pCodec)<0) {
        return _VRE_CREATE_CODEC_NOT_OPEN; // Could not open codec
    }
	
    return _VR_NO_ERROR;
}

int CVideoReader::InitFrameSize( int p_iWidth, int p_iHeight ) {
	
    m_iFrameWidth = m_pCodecCtx->width;
    m_iFrameHeight = m_pCodecCtx->height;
    
    AVRational timeBase = m_pFormatCtx->streams[m_iVideoStream]->time_base;
    m_dFrameTime2Sec = (double)timeBase.num / (double)timeBase.den;
    
    m_iTotalNumFrames = m_pFormatCtx->streams[m_iVideoStream]->nb_frames;
	
	// Allocate video frame
    m_pFrame=avcodec_alloc_frame();
	
    
    m_iWidth = p_iWidth;
    m_iHeight = p_iHeight;
    
    // Allocate RGB picture
	m_pFrameRGB = (AVPicture*) malloc (sizeof(AVPicture));
	
    avpicture_alloc(m_pFrameRGB, PIX_FMT_RGB24, p_iWidth, p_iHeight);
	m_bPictureAllocated = true;
    
    // Setup scaler
    static int sws_flags =  SWS_BICUBIC;
    m_pImg_convert_ctx = sws_getContext(m_pCodecCtx->width, 
										m_pCodecCtx->height,
										m_pCodecCtx->pix_fmt,
										p_iWidth, 
										p_iHeight,
										PIX_FMT_RGB24,
										sws_flags, NULL, NULL, NULL);
	
	if ( m_iTotalNumFrames != 0) {
		m_iInterFrameTime = m_pFormatCtx->streams[m_iVideoStream]->nb_frames / m_pFormatCtx->streams[m_iVideoStream]->duration;
		NextFrame();
	} else {
		
		NextFrame ();	
		int temp = m_iFrameTime;
		NextFrame ();
		
		m_iInterFrameTime = m_iFrameTime - temp;
		
		Restard   ();
		
		m_iTotalNumFrames = 0;
	}
    
    return _VR_NO_ERROR;
}


bool CVideoReader::NextFrame( void ) {
    AVPacket        packet;
    int             frameFinished = 0;
    int             avRead = 0;
	
    avRead = av_read_frame(m_pFormatCtx, &packet);
    while(!frameFinished && avRead >= 0) {
        // Is this a packet from the video stream?
        if(packet.stream_index==m_iVideoStream) {  
			//	avcodec_get_frame_defaults(m_pFrame);
            avcodec_decode_video2(m_pCodecCtx, m_pFrame, &frameFinished, &packet); 
            if ( frameFinished ) {
                m_iFrameTime = packet.dts;   
				av_free_packet(&packet);
				break;
            }            
        }
		
        // Free the packet that was allocated by av_read_frame
        av_free_packet(&packet);
        avRead = av_read_frame(m_pFormatCtx, &packet);
    }
	
    m_iSecond = (double) m_iFrameTime * m_dFrameTime2Sec;   
	
    if (avRead < 0) {
        return false;
    } else {
        m_bFrameConverted = false;
        ++m_iFrameNumber;
		
		if ( m_iTotalNumFrames < m_iFrameNumber)
			m_iTotalNumFrames = m_iFrameNumber;
		
        return true;  
    }
}


bool CVideoReader::GoToFrame ( int p_iFrameNum ) {
    if (p_iFrameNum == m_iFrameNumber) return true;
    if (p_iFrameNum == m_iFrameNumber + 1) return NextFrame();
	if ( (p_iFrameNum - m_iFrameNumber) < 15 && (p_iFrameNum - m_iFrameNumber) > 0 ) {
		for ( int i = 0; i < (p_iFrameNum - m_iFrameNumber) + 1; ++i ) {
			if (!NextFrame())
				return false;
		}
		
		return true;
	}
	
    return GoToTime((int64_t) p_iFrameNum * m_iInterFrameTime);
}

bool CVideoReader::GoToSecond ( double p_dSecond ) {
    return GoToTime((int64_t) p_dSecond / m_dFrameTime2Sec);
}

bool CVideoReader::GoToTime ( int64_t p_Time ) {
	//avformat_seek_file(ic, -1, INT64_MIN, timestamp, INT64_MAX, 0);

	avformat_seek_file(m_pFormatCtx, m_iVideoStream, p_Time, p_Time, p_Time, AVSEEK_FLAG_BACKWARD);
    
    AVPacket        packet;
    int             frameFinished = 0;
    int             avRead = 0;
    
    avRead = av_read_frame(m_pFormatCtx, &packet);
    while(avRead >= 0) {
        // Is this a packet from the video stream?
        if(packet.stream_index==m_iVideoStream) {          
            avcodec_decode_video2(m_pCodecCtx, m_pFrame, &frameFinished, &packet);   
            if ( frameFinished ) {			
                if (p_Time <= packet.dts) {
                    av_free_packet(&packet);
                    m_iSecond = (double) p_Time * m_dFrameTime2Sec;
                    m_iFrameTime = p_Time;
                    m_iFrameNumber = (int) p_Time / m_iInterFrameTime;
                    m_bFrameConverted = false;
                    
                    //printf("%d - %lld - %lld\n", m_iFrameNumber, packet.dts, m_iFrameTime); 
                    return true;
                }
            }            
        }
        
        // Free the packet that was allocated by av_read_frame
        av_free_packet(&packet);
        avRead = av_read_frame(m_pFormatCtx, &packet);
    }
	
   
    return false;
}

int CVideoReader::ConvertFrameToRGB( void ) {
	
    sws_scale (m_pImg_convert_ctx, m_pFrame->data, m_pFrame->linesize,
			   0, m_iFrameHeight,
			   m_pFrameRGB->data, m_pFrameRGB->linesize);	   
    
    m_bFrameConverted = true;
    
    return _VR_NO_ERROR;
    
}


VR_RESULT CVideoReader::GetFrame ( AVPicture** p_pFrameRGB ) {
    if (!m_bFrameConverted)
        ConvertFrameToRGB();          
    
    *p_pFrameRGB = m_pFrameRGB;
    
    return _VR_NO_ERROR;
}

VR_RESULT CVideoReader::GetFrame ( unsigned char** p_ucData, int* p_iWidth, int* p_iHeight, int* p_iLineSize ){
    if (!m_bFrameConverted)
        ConvertFrameToRGB();   
    
    *p_ucData = (unsigned char*)m_pFrameRGB->data[0];
    *p_iWidth = m_iWidth;
    *p_iHeight = m_iHeight;
    *p_iLineSize = m_pFrameRGB->linesize[0];
    
    return _VR_NO_ERROR;
}

int CVideoReader::GetWidth() {
    return m_iWidth;
}

int CVideoReader::GetHeight() {
    return m_iHeight;
    
}

int CVideoReader::GetFrameWidth() {
    return m_iFrameWidth;
}

int CVideoReader::GetFrameHeight() {
    return m_iFrameHeight;
    
}

int CVideoReader::GetFrameNumber() {
    return m_iFrameNumber;
}

CVideoReader::~CVideoReader( void ) {
    // Release old picture and scaler
	if ( m_bPictureAllocated )
		avpicture_free(m_pFrameRGB);
	
	free(m_pFrameRGB);
	
	if ( m_pImg_convert_ctx )
		sws_freeContext(m_pImg_convert_ctx);	
    
    // Free the YUV frame
	if ( m_pFrame )
		av_free(m_pFrame);
    
    // Close the codec
	if ( m_pCodecCtx )
		avcodec_close(m_pCodecCtx);
    
    // Close the video file
	if ( m_pFormatCtx )
		av_close_input_file(m_pFormatCtx);
    
    free(m_sFileName);
	// free(m_pBuf);
}

int64_t CVideoReader::GetTime ( void ) {
    return m_iFrameTime;
}

int CVideoReader::GetNumFrames() {
    return m_iTotalNumFrames;
}

int CVideoReader::GetSeconds() {
    return m_iSecond;
}


