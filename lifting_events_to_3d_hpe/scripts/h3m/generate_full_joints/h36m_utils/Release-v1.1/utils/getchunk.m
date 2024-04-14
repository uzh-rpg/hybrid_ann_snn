function chunkind = getchunk(ind,chunk,num_chunks)
	maxind = length(ind);
	chunklen = ceil(maxind/num_chunks);
	chunkind = ind((chunk-1)*chunklen+1 : min(chunk*chunklen,maxind));