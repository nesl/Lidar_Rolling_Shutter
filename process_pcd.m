%Load three different sensor readings

%bag = rosbag('lidar_audio/breath_julian.bag');
bag2 = rosbag('lidar_audio/150.bag');
bag3 = rosbag('lidar_audio/400.bag');

%pointBag = select(bag,'Topic','/os_cloud_node/points');
pointBag2 = select(bag2,'Topic','/os_cloud_node/points');
pointBag3 = select(bag3,'Topic','/os_cloud_node/points');


%pcMsgs = readMessages(pointBag);
pcMsgs2 = readMessages(pointBag2);
pcMsgs3 = readMessages(pointBag3);


%%


lens = 0;
%pcMsgsA = pcMsgs2;

for j=100:50:500
    
num = 0;
folder = strcat('data/', string(j), '/');
mkdir(folder);
    
%j = 100;
stream = [];

bag = rosbag(strcat('lidar_audio/', string(j), '.bag'));
pointBag = select(bag,'Topic','/os_cloud_node/points');
pcMsgsA = readMessages(pointBag);

%{

for i=1:length(pcMsgsA)
    MsgsTemp = pcMsgsA{i}; %Choose a specific sensor reading: pcMsgs#
    MsgsTemp.PreserveStructureOnRead = true;
    intensityTemp = readField(Msgs, 'intensity');
    if i == 1
        intensityAcc = intensityTemp;
    else
        intensityAcc = intensityAcc + intensityTemp;
    end
end
intensityAvg = intensityAcc/length(pcMsgsA);
%}

for i=1:length(pcMsgsA)
     %i = 80; %Choose a specific scan
     
     Msgs = pcMsgsA{i}; %Choose a specific sensor reading: pcMsgs#
     Msgs.PreserveStructureOnRead = true;

     pc = readXYZ(Msgs); %Not needed
     intensity = readField(Msgs, 'intensity');
     otherField = readField(Msgs, 'reflectivity'); %If we want to use another field
     intensity2 = intensity;
     %intensity = intensity - double(otherField);
     intensity(intensity2 < 2000) = 0; %Filter out points with intensity lower than 2000
     

     otherField = otherField .* uint16(intensity > 0); %Filter out points according to intensity
     otherField( ~any(otherField,2), : ) = [];
     otherField( :, ~any(otherField,1) ) = []; %Reduce sparsity of matrix
     otherField_minus_mean = double(otherField) - mean(otherField,2);
     
     %{
     intensityAvg2 = intensityAvg .* double(intensity > 0);
     intensityAvg2( ~any(intensityAvg2,2), : ) = [];
     intensityAvg2( :, ~any(intensityAvg2,1) ) = []; %Reduce sparsity of matrix
     %}
     
     
     intensity( ~any(intensity,2), : ) = [];
     %intensity(intensity == 0) = NaN;
     intensity( :, ~any(intensity,1) ) = []; %Reduce sparsity of matrix
     %intensity = intensity(:,1:43); %Remove certain values because there
     %is a gap in the readings?
     intensity_minus_mean = intensity - mean(intensity,2);%mean(intensity(intensity > 0)); %Subtract mean
     %intensity_minus_mean = intensity - intensityAvg2;
     %intensity_minus_mean = mean(intensity_minus_mean,1);
     
     %intensity = intensity(:,1:43)
     %intensity_minus_mean = rmoutliers(intensity_minus_mean);
     %{
     for j=size(intensity_minus_mean,1)
         intensity_minus_mean(j,:) = intensity_minus_mean(j,:)/sqrt(sum(abs(intensity_minus_mean(j,:) .^2))/length(intensity_minus_mean(j,:)));
     end
     %}
     %intensity_minus_mean = normalize(intensity_minus_mean);
          
     %{ 
     %If we want to calculate the mean intensity based on all scans
     if(i == 1)
        acc_intensity = intensity;
        intensity_minus_mean = intensity;
     else
        acc_intensity = acc_intensity + intensity;
        intensity_minus_mean = intensity - acc_intensity/i;
     end
     %}
     %{
     imagesc(intensity_minus_mean)
     for j=1:size(intensity_minus_mean,1)
        plot((1:size(intensity_minus_mean,2))*0.1/2048, intensity_minus_mean(j,:),'DisplayName',string(j))
        hold on
     end
     legend
     %}
     %figure

     fs = 20480;
     
     %S = reshape(intensity_minus_mean, [], 1); %If we want to fuse all
     %rows into a single series of readings
     %S = otherField_minus_mean; %intensity_minus_mean;
     S = intensity_minus_mean;
     
     %%{
     if size(S,1) == 14 && size(S,2) >= 70
        %stream = [stream S];
        writematrix(S,strcat(folder, string(num),'.csv'),'Delimiter',',');
        num = num + 1;
        length(S)
        if lens == 0 || length(S) < lens
            lens = length(S);
        end
     end
     %}
     
     %{
     %for j=1:size(intensity_minus_mean,1)
         %S = intensity_minus_mean(8,:);
         n = length(S);
         X = fft(S);
         Y = fftshift(X);
         fshift = (-n/2:n/2-1)*(fs/n); % zero-centered frequency range
         powershift = abs(Y).^2/n;     % zero-centered power
         figure
         plot(fshift,powershift)
         %imagesc(intensity_minus_mean) %See the signal as a 2D image
         legend
         %{
         tmp_val = fshift(find(powershift == max(powershift)));
         all_scans(i,j) = min(abs(tmp_val));
         %}
     %end
     %}
     %t = 0:1/fs:((length(stream)-1)/fs);
     

 
end 

%writematrix(stream,strcat(string(j),'.csv'),'Delimiter',',');

end


%% Player - Watch in Real-Time

player = pcplayer([-5 5],[-5 5],[-5 5]);

while isOpen(player) 
    for i=1:length(pcMsgs)
         Msgs = pcMsgs{i};
         Msgs.PreserveStructureOnRead = true;
         pc = readXYZ(Msgs);
         intensity = readField(Msgs, 'intensity');
         %intensity(intensity < 2000) = 0;
         %{
         intensity( ~any(intensity,2), : ) = [];
         intensity( :, ~any(intensity,1) ) = [];
         %imagesc(intensity - mean(intensity(intensity > 0)))
         idxs = find(intensity > 2000);
         pc = pc(idxs,:);
         intensity = intensity(idxs);
         %}
         ptCloud = pointCloud(pc,'Intensity', intensity);
         view(player,ptCloud);    
    end 
end

%% Breathing processing

all_scans = zeros(100,14);
acc_intensity = [];
for i=1:length(pcMsgs)
     %i = 65;
     subplot(2,1,1)
     Msgs = pcMsgs{i};
     Msgs.PreserveStructureOnRead = true;

     %Segment point cloud based on euclidean distance and select manually
     %the person
     pc = readXYZ(Msgs);
     ptCloud = pointCloud(pc);
     [labels, numClusters] = pcsegdist(ptCloud, 0.5);
     labels(labels ~= 5) = 0;
     pcshow(ptCloud.Location, labels);
     
     newptCloud = pc(:,:,3) .* double(labels > 0); %xyz?? or other property
     newptCloud( ~any(newptCloud,2), : ) = [];
     newptCloud( :, ~any(newptCloud,1) ) = [];
     
     intensity_minus_mean = newptCloud - mean(newptCloud,2);%mean(intensity(intensity > 0));
  
     fs = 20480;
     
     %S = reshape(intensity_minus_mean, [], 1);
     %S = otherField_minus_mean; %intensity_minus_mean;
     S = intensity_minus_mean;
     
     %for j=1:size(intensity_minus_mean,1)
         %S = intensity_minus_mean(8,:);
         n = length(S);
         X = fft(S);
         Y = fftshift(X);
         fshift = (-n/2:n/2-1)*(fs/n); % zero-centered frequency range
         powershift = abs(Y).^2/n;     % zero-centered power
         subplot(2,1,2)
         plot(fshift,powershift)
         legend
         %{
         tmp_val = fshift(find(powershift == max(powershift)));
         all_scans(i,j) = min(abs(tmp_val));
         %}
     %end
     
     %{
     intensity( ~any(intensity,2), : ) = [];
     intensity( :, ~any(intensity,1) ) = [];
     imagesc(intensity - mean(intensity(intensity > 0)))
     idxs = find(intensity > 2000);
     pc = pc(idxs,:);
     intensity = intensity(idxs);
     %}
     %ptCloud = pointCloud(pc,'Intensity', intensity2);
 
end 

%%
pc = readXYZ(pcMsgs2{100});
intensity = readField(pcMsgs2{100}, 'reflectivity');
pcshow(pc, intensity);

%{
ptCloud = pcread('pcdfiles_220Hz/2064.308731940.pcd');
pcshow(ptCloud);
%}