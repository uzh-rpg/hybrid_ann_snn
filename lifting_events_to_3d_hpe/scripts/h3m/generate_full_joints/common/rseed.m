function rseed()

s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s);