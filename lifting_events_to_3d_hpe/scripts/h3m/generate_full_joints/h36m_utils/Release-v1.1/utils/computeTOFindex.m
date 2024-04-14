function [mapping,indicator] = computeTOFindex(NFrames, times)
    mapping = zeros(1,NFrames);
    times = times - times(end);
    times = double(times) * 50.0/1000.0;
    for i=length(times):-1:1
        j = NFrames + times(i);
        rj = round(j);
        if(rj<2)
            %disp(['end frames. ' num2str(i)]);
            break;
        end
        if(mapping(rj)==0)
            mapping(rj) = i;
        else
            otherj = NFrames + times(mapping(rj));
            if(abs(j-rj) < abs(otherj-rj))
            	mapping(rj) = i;
            end
        end
    end
    indicator = mapping>0;
    k = round(NFrames+times(i+1));
    for j=1:k-1
        mapping(j)=mapping(k);
    end
    changes = 1;
    while(changes)
        changes = 0;
        mapping2 = mapping;
        for j=2:NFrames-1
            if(mapping(j)==0)
                changes = 1;
                if(mapping(j-1)>0 && mapping(j+1)>0)
                    a1 = NFrames + times(mapping(j-1));
                    a2 = NFrames + times(mapping(j+1));
                    if(abs(a1-j) <= abs(a2-j))
                        mapping2(j) = mapping(j-1);
                    else
                        mapping2(j) = mapping(j+1);
                    end
                elseif(mapping(j-1)>0 && mapping(j+1)==0)
                    mapping2(j)=mapping(j-1);
                elseif(mapping(j-1)==0 && mapping(j+1)>0)
                    mapping2(j)=mapping(j+1);
                end
            end
        end
        mapping = mapping2;
    end
end