clc

if (ispc)
    mex -I./ -I../../Bin/3rdParty/ffmpeg-win64/include -L../../Bin/3rdParty/ffmpeg-win64/lib -lavcodec -lavdevice -lavformat -lavutil -lswscale mexVideoReader.cpp VideoReader.cpp
end

if (ismac)
else
    if (isunix)
        mex -I./ -I/usr/include -L/usr/lib -lavcodec -lavdevice -lavformat -lavutil -lswscale mexVideoReader.cpp VideoReader.cpp
    end
end