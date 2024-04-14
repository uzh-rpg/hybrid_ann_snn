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


#include "VideoWriter.h"
#include <string.h>
#include <time.h>



#define _VW_NO_ERROR													0

#define _VWE_CONSTRUCTOR_COULD_NOT_FIND_FORMAT							-1
#define _VWE_CONSTRUCTOR_MEMORY_ERROR									-2
#define _VWE_CONSTRUCTOR_INVALID_OUTPUT_FORMAT_PARAM					-3
#define _VWE_CONSTRUCTOR_COULD_NOT_OPEN_FILE							-4
#define _VWE_ADD_FRAME_RGB_CAN_NOT_CONVERT								-5
#define _VWE_ADD_VIDEO_STREAM_NOT_ALLOC									-6
#define _VWE_OPEN_VIDEO_CODEC_NOT_FOUND									-7
#define _VWE_OPEN_VIDEO_COULD_NOT_OPEN_CODEC							-8
#define _VWE_OPEN_VIDEO_COULD_NOT_ALLOCATE_PICTURE						-9
#define _VWE_OPEN_VIDEO_COULD_NOT_ALLOCATE_PIC_BUF						-10
#define _VWE_WRITE_VIDEO_FRAME_ERROR_WHILE_WRITING						-11
#define _VWE_UKNOWN_ERROR												-12

#define _VWE_TEXT_CONSTRUCTOR_COULD_NOT_FIND_FORMAT						"Could not find suitable output format"
#define _VWE_TEXT_CONSTRUCTOR_MEMORY_ERROR								"Memory error"
#define _VWE_TEXT_CONSTRUCTOR_INVALID_OUTPUT_FORMAT_PARAM				"Invalid output format parameters"
#define _VWE_TEXT_CONSTRUCTOR_COULD_NOT_OPEN_FILE						"Could not open file"
#define _VWE_TEXT_ADD_FRAME_RGB_CAN_NOT_CONVERT							"Cannot initialize the conversion context"
#define _VWE_TEXT_ADD_VIDEO_STREAM_NOT_ALLOC							"Could not alloc stream"
#define _VWE_TEXT_OPEN_VIDEO_CODEC_NOT_FOUND							"Codec not found"
#define _VWE_TEXT_OPEN_VIDEO_COULD_NOT_OPEN_CODEC						"Could not open codec"
#define _VWE_TEXT_OPEN_VIDEO_COULD_NOT_ALLOCATE_PICTURE					"Could not allocate picture"
#define _VWE_TEXT_OPEN_VIDEO_COULD_NOT_ALLOCATE_PIC_BUF					"Could not allocate picture buffer"
#define _VWE_TEXT_WRITE_VIDEO_FRAME_ERROR_WHILE_WRITING					"Error while writing video frame"
#define _VWE_TEXT_UKNOWN_ERROR											"Uknown error"


#define VW_CHECK(x)						{	VW_RESULT tempRes = x; \
											if ( tempRes < 0 ) return tempRes; }

CVideoWriter::CVideoWriter( const char* p_sFileName, int p_iWidth, int p_iHeight, int p_iFps, int p_iBitRate, char* p_pFormat, VW_RESULT* p_iResult ) {	
    
	m_sFileName			= NULL;
    m_pFmt				= NULL;
    m_pOc				= NULL;		
	m_pPicture			= NULL;
	m_ucVideo_outbuf	= NULL;
	m_pVideo_st			= NULL;
	
	m_sFileName = (char*) malloc (sizeof(char) * (strlen(p_sFileName) + 1));
    strcpy(m_sFileName, p_sFileName);  
		
	m_iWidth	= p_iWidth;
	m_iHeight	= p_iHeight;
	m_iFps		= p_iFps;
	
	m_iBitRate = p_iBitRate;
	
    /* initialize libavcodec, and register all codecs and formats */
    av_register_all();
	
    /* auto detect the output format from the name. default is
	 mpeg. */
    //m_pFmt = av_guess_format("h264" , NULL, "video/mp4");
	m_pFmt = av_guess_format(NULL, m_sFileName, NULL);
    if (!m_pFmt) {
        m_pFmt = av_guess_format("mpeg", NULL, NULL);
    }
    if (!m_pFmt) {
		*p_iResult = _VWE_CONSTRUCTOR_COULD_NOT_FIND_FORMAT;
		
        return;
    }
	
	strcpy( p_pFormat, m_pFmt->long_name );
	
    /* allocate the output media context */
    m_pOc = avformat_alloc_context();
    if (!m_pOc) {
        *p_iResult = _VWE_CONSTRUCTOR_MEMORY_ERROR;
        return;
    }
    m_pOc->oformat = m_pFmt;
    snprintf(m_pOc->filename, sizeof(m_pOc->filename), "%s", m_sFileName);
	
    /* add the audio and video streams using the default format codecs
	 and initialize the codecs */
    m_pVideo_st = NULL;

    if (m_pFmt->video_codec != CODEC_ID_NONE) {
        *p_iResult = AddVideoStream();
		
		if ( *p_iResult < 0 ) return;
    }
	
    /* set the output parameters (must be done even if no
	 parameters). */
    if (av_set_parameters(m_pOc, NULL) < 0) {
        *p_iResult = _VWE_CONSTRUCTOR_INVALID_OUTPUT_FORMAT_PARAM;
        return;
    }
	
    dump_format(m_pOc, 0, m_sFileName, 1);
	
    /* now that all the parameters are set, we can open the audio and
	 video codecs and allocate the necessary encode buffers */
    if (m_pVideo_st) {
        *p_iResult = OpenVideo();
		
		if ( *p_iResult < 0 ) return;
	}
	
    /* open the output file, if needed */
    if (!(m_pFmt->flags & AVFMT_NOFILE)) {
        if (url_fopen(&m_pOc->pb, m_sFileName, URL_WRONLY) < 0) {
            *p_iResult = _VWE_CONSTRUCTOR_COULD_NOT_OPEN_FILE;
            return;
        }
    }
	
    /* write the stream header, if any */
    av_write_header(m_pOc);
}

CVideoWriter::~CVideoWriter( void ) {
	/* write the trailer, if any.  the trailer must be written
     * before you close the CodecContexts open when you wrote the
     * header; otherwise write_trailer may try to use memory that
     * was freed on av_codec_close() */
    av_write_trailer(m_pOc);
	
    /* close each codec */
    if (m_pVideo_st)
        CloseVideo();
	
    /* free the streams */
	if ( m_pOc ) {
		for(int i = 0; i < m_pOc->nb_streams; ++i) {
			av_freep(&m_pOc->streams[i]->codec);
			av_freep(&m_pOc->streams[i]);
		}
		
		if (!(m_pFmt->flags & AVFMT_NOFILE)) {
			/* close the output file */
			url_fclose(m_pOc->pb);
		}
		
		/* free the stream */
		av_free( m_pOc );	
	}
	
	if ( m_sFileName )
		free( m_sFileName );
	
}

VW_RESULT CVideoWriter::AddFrameRGB24 ( AVPicture* p_pPicture, int p_iWidth, int p_iHeight ) {
	struct SwsContext *img_convert_ctx;
    
    // Setup scaler
    img_convert_ctx = sws_getContext(   p_iWidth, 
										p_iHeight,
										PIX_FMT_RGB24,
										m_pVideo_st->codec->width, 
										m_pVideo_st->codec->height,
										m_pVideo_st->codec->pix_fmt,
										SWS_BICUBIC, NULL, NULL, NULL	);
	
	if (img_convert_ctx == NULL) {
		return _VWE_ADD_FRAME_RGB_CAN_NOT_CONVERT;
	}
	
	sws_scale ( img_convert_ctx, p_pPicture->data, p_pPicture->linesize, 0, p_iHeight, m_pPicture->data, m_pPicture->linesize );	 
	
	VW_CHECK( WriteVideoFrame() );
	
	sws_freeContext(img_convert_ctx);
	
	return _VW_NO_ERROR;
}

const char* CVideoWriter::TranslateError ( VW_RESULT error ) {
	
	switch ((int) error ) {
		case _VWE_CONSTRUCTOR_COULD_NOT_FIND_FORMAT:
			return _VWE_TEXT_CONSTRUCTOR_COULD_NOT_FIND_FORMAT;
		case _VWE_CONSTRUCTOR_MEMORY_ERROR:
			return _VWE_TEXT_CONSTRUCTOR_MEMORY_ERROR;
		case _VWE_CONSTRUCTOR_INVALID_OUTPUT_FORMAT_PARAM:
			return _VWE_TEXT_CONSTRUCTOR_INVALID_OUTPUT_FORMAT_PARAM;
		case _VWE_CONSTRUCTOR_COULD_NOT_OPEN_FILE:
			return _VWE_TEXT_CONSTRUCTOR_COULD_NOT_OPEN_FILE;	
		case _VWE_ADD_FRAME_RGB_CAN_NOT_CONVERT:
			return _VWE_TEXT_ADD_FRAME_RGB_CAN_NOT_CONVERT;			
		case _VWE_ADD_VIDEO_STREAM_NOT_ALLOC:
			return _VWE_TEXT_ADD_VIDEO_STREAM_NOT_ALLOC;				
		case _VWE_OPEN_VIDEO_CODEC_NOT_FOUND:
			return _VWE_TEXT_OPEN_VIDEO_CODEC_NOT_FOUND;		
		case _VWE_OPEN_VIDEO_COULD_NOT_OPEN_CODEC:
			return _VWE_TEXT_OPEN_VIDEO_COULD_NOT_OPEN_CODEC;
		case _VWE_OPEN_VIDEO_COULD_NOT_ALLOCATE_PICTURE:
			return _VWE_TEXT_OPEN_VIDEO_COULD_NOT_ALLOCATE_PICTURE;
		case _VWE_OPEN_VIDEO_COULD_NOT_ALLOCATE_PIC_BUF:
			return _VWE_TEXT_OPEN_VIDEO_COULD_NOT_ALLOCATE_PIC_BUF;	
		case _VWE_WRITE_VIDEO_FRAME_ERROR_WHILE_WRITING:	
			return _VWE_TEXT_WRITE_VIDEO_FRAME_ERROR_WHILE_WRITING;
		case _VWE_UKNOWN_ERROR:							
		default:
			return _VWE_TEXT_UKNOWN_ERROR;
	};
}


/**************************************************************/
/* video output */

/* add a video output stream */
VW_RESULT CVideoWriter::AddVideoStream( void )
{
    AVCodecContext *c;
	
    m_pVideo_st = av_new_stream(m_pOc, 0);
    if (!m_pVideo_st) {
        return _VWE_ADD_VIDEO_STREAM_NOT_ALLOC;
    }
	
    c = m_pVideo_st->codec;
    c->codec_id = m_pFmt->video_codec;
    c->codec_type = AVMEDIA_TYPE_VIDEO;
	
    /* put sample parameters */
    c->bit_rate = m_iBitRate;
    /* resolution must be a multiple of two */
    c->width = m_iWidth;
    c->height = m_iHeight;
    /* time base: this is the fundamental unit of time (in seconds) in terms
	 of which frame timestamps are represented. for fixed-fps content,
	 timebase should be 1/framerate and timestamp increments should be
	 identically 1. */
    c->time_base.den = m_iFps;
    c->time_base.num = 1;
    c->gop_size = 12; /* emit one intra frame every twelve frames at most */
    c->pix_fmt = PIX_FMT_YUV420P;
	//c->pix_fmt = PIX_FMT_RGB32;
	
    if (c->codec_id == CODEC_ID_MPEG2VIDEO) {
        /* just for testing, we also add B frames */
        c->max_b_frames = 2;
    }
    if (c->codec_id == CODEC_ID_MPEG1VIDEO){
        /* Needed to avoid using macroblocks in which some coeffs overflow.
		 This does not happen with normal video, it just happens here as
		 the motion of the chroma plane does not match the luma plane. */
        c->mb_decision = 2;
    }
    // some formats want stream headers to be separate
    if(m_pOc->oformat->flags & AVFMT_GLOBALHEADER)
        c->flags |= CODEC_FLAG_GLOBAL_HEADER;
	
    return _VW_NO_ERROR;
}

VW_RESULT CVideoWriter::OpenVideo( void )
{
    AVCodec *codec;
    AVCodecContext *c;
	
    c = m_pVideo_st->codec;
	
    /* find the video encoder */
    codec = avcodec_find_encoder(c->codec_id);
    if (!codec) {
        return _VWE_OPEN_VIDEO_CODEC_NOT_FOUND;
    }
	
    /* open the codec */
    if (avcodec_open(c, codec) < 0) {
        return _VWE_OPEN_VIDEO_COULD_NOT_OPEN_CODEC;
    }
	
    m_ucVideo_outbuf = NULL;
    if (!(m_pOc->oformat->flags & AVFMT_RAWPICTURE)) {
        /* allocate output buffer */
        /* XXX: API change will be done */
        /* buffers passed into lav* can be allocated any way you prefer,
		 as long as they're aligned enough for the architecture, and
		 they're freed appropriately (such as using av_free for buffers
		 allocated with av_malloc) */
        video_outbuf_size = 200000;
        m_ucVideo_outbuf = (uint8_t*)av_malloc(video_outbuf_size);
    }
	
    /* allocate the encoded raw m_pPicture */
	
	m_pPicture = avcodec_alloc_frame();
	
    if (!m_pPicture)
        return _VWE_OPEN_VIDEO_COULD_NOT_ALLOCATE_PICTURE;
	
    int size = avpicture_get_size(c->pix_fmt, c->width, c->height);
    uint8_t* picture_buf = (uint8_t*)av_malloc(size);
	
    if (!picture_buf) {
        av_free(m_pPicture);
        return _VWE_OPEN_VIDEO_COULD_NOT_ALLOCATE_PIC_BUF;
    }
	
    avpicture_fill((AVPicture *)m_pPicture, picture_buf, c->pix_fmt, c->width, c->height);
	
	return _VW_NO_ERROR;
}

VW_RESULT CVideoWriter::WriteVideoFrame( void )
{
    int out_size, ret;
    AVCodecContext *c;
	
    c = m_pVideo_st->codec;
	
    if (m_pOc->oformat->flags & AVFMT_RAWPICTURE) {
        /* raw video case. The API will change slightly in the near
		 futur for that */
        AVPacket pkt;
        av_init_packet(&pkt);
		
        pkt.flags |= AV_PKT_FLAG_KEY;
        pkt.stream_index= m_pVideo_st->index;
        pkt.data= (uint8_t *)&m_pPicture;
        pkt.size= sizeof(AVPicture);
		
        ret = av_interleaved_write_frame(m_pOc, &pkt);
    } else {
        /* encode the image */
        out_size = avcodec_encode_video(c, m_ucVideo_outbuf, video_outbuf_size, m_pPicture);
        /* if zero size, it means the image was buffered */
        if (out_size > 0) {
            AVPacket pkt;
            av_init_packet(&pkt);
			
            if (c->coded_frame->pts != AV_NOPTS_VALUE)
                pkt.pts= av_rescale_q(c->coded_frame->pts, c->time_base, m_pVideo_st->time_base);
            if(c->coded_frame->key_frame)
                pkt.flags |= AV_PKT_FLAG_KEY;
            pkt.stream_index= m_pVideo_st->index;
            pkt.data= m_ucVideo_outbuf;
            pkt.size= out_size;
			
            /* write the compressed frame in the media file */
            ret = av_interleaved_write_frame(m_pOc, &pkt);
        } else {
            ret = 0;
        }
    }
    if (ret != 0) {
        return _VWE_WRITE_VIDEO_FRAME_ERROR_WHILE_WRITING;
    }
    m_iFrame_count++;
	
	return _VW_NO_ERROR;
}

VW_RESULT CVideoWriter::CloseVideo( void )
{	
    avcodec_close( m_pVideo_st->codec );
	
	if ( m_pPicture ) {
		av_free( m_pPicture->data[0] );
		av_free( m_pPicture );
	}
	
	if ( m_ucVideo_outbuf ) 
		av_free( m_ucVideo_outbuf );
	
	return _VW_NO_ERROR;
}






