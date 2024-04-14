function F = H36MComputeFeatures(Sequence, Features)
% DB4MPComputeFeatures compute or load all features corresponding to a
% sequence. The reading contains synchronization of the features independent
% of the modality from which it was obtained (image, mask, pose, depth).

%% compute feature dependency graph and check what is precomputed
[AllFeatures IndexRequiredFeatures IndexFeatures AllFeaturesAccess] = updateFeatureList(Sequence, Features);

%% compute one by one the intermediate and then final features
Subject = Sequence.getSubject();
Camera  = Sequence.getCamera();

idx = Sequence.getIdxFrames();

for i = 1: length(Features)
    if ~isa(Features{i},'H36MVideoFeature')
        F{i} = zeros(length(idx),Features{i}.getFeatureSize,'single');
    end
end

t = tic;
for f = 1: Sequence.getNumFrames
    for i = IndexFeatures
        if IndexRequiredFeatures{i}==Inf
            % this one we load
            [Ftmp{i} AllFeaturesAccess{i}] = AllFeaturesAccess{i}.getFrame(idx(f));
        else
            % this one we compute
            Ftmp{i} = AllFeatures{i}.process(Ftmp{IndexRequiredFeatures{i}},Subject,Camera);
            
            % add to save buffer
            AllFeaturesAccess{i} = AllFeaturesAccess{i}.putFrame(idx(f),Ftmp{i});
        end
        
        if i < length(Features)+1
            if isa(Features{i},'H36MVideoFeature')
                F{i}{f} = Ftmp{i};
            else
                F{i}(f,:) = Ftmp{i};
            end
        end
    end
end


% save precomputed features
if length(idx)==H36MDataBase.instance.getNumFrames(Sequence.Subject,Sequence.Action,Sequence.SubAction)
    for i = 1: length(AllFeaturesAccess)
        AllFeaturesAccess{i}.save();
    end
end

% normalize the features that are going out
for i = 1: length(Features)
  if ~isa(Features{i},'H36MVideoFeature')
    F{i} = AllFeatures{i}.normalize(F{i},Subject,Camera);
  end
end

end

function [AllFeatures IndexRequiredFeatures IndexFeatures AllFeaturesAccess] = updateFeatureList(Sequence, Features)
IndexRequiredFeatures = cell(1,length(Features));

i = 1;
while i <= length(Features)
    AllFeaturesAccess{i} = Features{i}.serializer(Sequence);
    if AllFeaturesAccess{i}.exist()
        % Feature is precomputed so no children check necessary
        IndexRequiredFeatures{i} = Inf;
    else
        % not precomputed
        IndexRequiredFeatures{i} = [];
        if isempty(Features{i}.RequiredFeatures)
            errstring = 'This is an error because feature is not precomputed and cannot be computed from anything!';
            errstring = [errstring '\n' 'Required feature name: ' Features{i}.FeatureName];
            errstring = [errstring '\n' 'Required feature path: ' strrep(Features{i}.FeaturePath, '\', '\\')];
            errstring = [errstring '\n' 'Required feature extension: ' Features{i}.Extension];
            errstring = [errstring '\n' 'Sequence name: ' Sequence.BaseName];
            error('MATLAB:LoadErr', errstring);
        else
            % (required features and) the feature itself need to be computed
            for j = 1: length(Features{i}.RequiredFeatures)
                % check if the j-th required features is already in the list of features
                v = cellfun(@(x)strcmp(x.FeatureName,Features{i}.RequiredFeatures{j}.FeatureName),Features);
                if any(v)
                    % already in the list
                    IndexRequiredFeatures{i} = [IndexRequiredFeatures{i} find(v)];
                else
                    % add to the list
                    Features = [Features Features{i}.RequiredFeatures(j)];
                    IndexRequiredFeatures{i} = [IndexRequiredFeatures{i} length(Features)];
                end
            end
        end
    end
    
    
    % check if these are already in
    i = i + 1;
end

% 	IndexRequiredFeatures = IndexRequiredFeatures';
% find the order in which the features should be computed
% 	[~, IndexFeatures] = sort(cellfun(@(x)(x(1)),IndexRequiredFeatures),2,'ascend');
[~, IndexFeatures] = sort(cellfun(@(x)(min(x)),IndexRequiredFeatures),2,'descend');
AllFeatures = Features;
end
