
%y 0.8-1
%x -0.3 - 0.6

%%{

%bags = {'high', 'medium', 'low', 'low2', 'low3', 'low4', 'low5'};
%bags = {'1000', '950', '800', '700', '600', '500', '400', '300', '250', '150'};
%bags = {'1000', '900', '800', '700', '600', '500', '400', '300', '200', '100'};
bags = {'1000_', '900_', '800_', '700_', '600_', '500_', '400_', '300_'};
warning('off', 'MATLAB:print:ContentTypeImageSuggested');
warning('off', 'MATLAB:print:ExportExcludesUI');

intensity_factor = 40; %10; %50;

for f=1:length(bags)

bag2 = rosbag(strcat('/media/hdd/new_locations/', string(bags(f)), '5.bag'));

pointBag2 = select(bag2,'Topic','/os_cloud_node/points');


pcMsgs2 = readMessages(pointBag2);


folder = strcat('motion_images_locations/', string(bags(f)), '/');
mkdir(folder);

for j=1:length(pcMsgs2)

    Msgs = pcMsgs2{j};
    Msgs.PreserveStructureOnRead = true;

    pc = readXYZ(Msgs);

    intensity = readField(Msgs, 'intensity');
    range = readField(Msgs, 'range');
    x = pc(:,:,1);
    y = pc(:,:,2);
    z = pc(:,:,3);

    %x = (x > -0.3 & x < 0.6);
    %y = (y > 0.8 & y < 1);
    %y = (y > 0.15 & y < 0.9);
    %x = (x > -2.0 & x < -1.6);
    %y = (y > -1.3 & y < 0);
    %x = (x > -3.0 & x < -2.4);
    %z = (z > -0.4 & z < 0.4);
    
    
    x=(x > -1.1 & x < 0.4); 
    y=(y > 1.2 & y < 2.3); 
    z= (z > -0.4 & z < 0.3);


    r = x.*y.*z;

    intensity = intensity.*r;

    intensity(intensity > intensity_factor) = 2000;
    intensity(intensity <= intensity_factor) = 0;
    pc(:,:,1) = pc(:,:,1).*r;
    pc(:,:,2) = pc(:,:,2).*r;
    pc(:,:,3) = pc(:,:,3).*r;
    %set(gcf,'Visible','on') 
    %pcshow(pc, intensity);

    
    idx = 1;
    for i=1:size(intensity,1)
       if ~all(intensity(i, :) == 0)
           break
       end

       idx = i;
    end

    idx2 = 32;
    for i=size(intensity,1):-1:1
       if ~all(intensity(i, :) == 0)
           break
       end

       idx2 = i;
    end

    idx3 = 1;
    for i=1:size(intensity,2)
       if ~all(intensity(:,i) == 0)
           break
       end

       idx3 = i;
    end

    idx4 = 2048;
    for i=size(intensity,2):-1:1
       if ~all(intensity(:,i) == 0)
           break
       end

       idx4 = i;
    end

    %range(range > 100) = 0;
    %range(range > 0 ) = 2000;

    x = pc(:,:,1);
    y = pc(:,:,2);
    z = pc(:,:,3);

    x1 = x(idx:idx2, idx3:idx4);
    y1 = y(idx:idx2, idx3:idx4);
    z1 = z(idx:idx2, idx3:idx4);
    %tt = pointcloud2image(x1(:),y1(:),z1(:), 100,100);
    %imshow(tt);

    pc_new = zeros(size(x1,1),size(x1,2), 3);
    pc_new(:,:,1) = x1;
    pc_new(:,:,2) = y1;
    pc_new(:,:,3) = z1;
    intensity = intensity(idx:idx2, idx3:idx4);
    range = range(idx:idx2, idx3:idx4);

    %set(gcf,'Visible','on') 
    %pcshow(pc_new, range);
    
    
   
    %j
    %{
    if isempty(pc_new)
        j
        continue
    end
    %}
    
    if all(intensity==0, 'all')
        j
        continue
    end
    
    
    set(gcf,'Visible','off') 
    %set(gcf, 'Units', 'pixels')
    %set(gcf, 'Position', [1 1 300 300])
    output_size = [400 400];%Size in pixels
    resolution = 300;%Resolution in DPI
    set(gcf,'paperunits','inches','paperposition',[0 0 output_size/resolution]);

    pcshow(pc_new, intensity );
    set(gcf, 'color', 'black')
    set(gcf, 'InvertHardCopy', 'off')

    colormap('gray');
    grid off
    axis off
    %view([0,0])
    %view([80,0])
    %view([100,0])
    view([0,0])
    zoom(1)
    file_name = strcat(folder, string(j), '_5.jpg');
    %exportgraphics(gcf,file_name,'ContentType', 'vector', 'BackgroundColor','black');
    %exportgraphics(gca,file_name,'ContentType', 'image', 'BackgroundColor','black', 'Resolution', 300);
    print(gcf, file_name, '-djpeg', ['-r' num2str(resolution)] )
end
end
%}

%{
distThreshold = 0.5;
Msgs = pcMsgs2{208};
Msgs.PreserveStructureOnRead = true;
pc = readXYZ(Msgs);
ptCloudObj = pointCloud(pc);
[labels,numClusters] = segmentLidarData(ptCloudObj,distThreshold);
labels(labels ~= 177) = 0;
labels(labels == 177) = 2000;
pcshow(ptCloudObj.Location, labels)
%colormap([hsv(2);[0 0 0]])
%}

%%

bag2 = rosbag('/media/hdd/motion_position4/1000.bag');

pointBag2 = select(bag2,'Topic','/os_cloud_node/points');


pcMsgs2 = readMessages(pointBag2);

player = pcplayer([-5 5],[-5 5],[-5 5]);

while isOpen(player) 
    for i=1:length(pcMsgs2)
         Msgs = pcMsgs2{i};
         Msgs.PreserveStructureOnRead = true;
         pc = readXYZ(Msgs);
         intensity = readField(Msgs, 'intensity');
         ptCloud = pointCloud(pc,'Intensity', intensity);
         view(player,ptCloud);    
    end 
end
%%
bag2 = rosbag('/media/hdd/new_locations/1000_1.bag');

pointBag2 = select(bag2,'Topic','/os_cloud_node/points');


pcMsgs2 = readMessages(pointBag2);

%%

Msgs = pcMsgs2{4};
Msgs.PreserveStructureOnRead = true;
pc = readXYZ(Msgs);
intensity = readField(Msgs, 'intensity');
intensity_factor = 40; %10;



x = pc(:,:,1);
y = pc(:,:,2);
z = pc(:,:,3);

%y = (y > 0.15 & y < 0.9);
%y = (y > -1.3 & y < 0);
%x = (x > -2.0 & x < -1.6);
%x = (x > -3.0 & x < -2.4);
%z = (z > -0.4 & z < 0.4);

%y = (y > 1 & y < 2);
%x = (x > -1.9 & x < -0.5);
%z = (z > -0.3 & z < 0.3);

%_1 x=(x > -2.8 & x < -2.3);  y= (y > -0.5 & y < 0.9); z= (z > -0.4 & z < 0.3); view 90, thresh 40
%_2 x=(x > -2.4 & x < -1.4); y= (y > -2 & y < -1); z= (z > -0.4 & z < 0.3);
%view 130 thresh 40
%_3 x=(x > -2.9 & x < -1.6); y= (y > -3.6 & y < -2.7); z= (z > -0.4 & z <
%0.3); view 150 thresh 40
%_4 x=(x > -1.9 & x < -0.7); y= (y > -1.3 & y < 0.1); z= (z > -0.4 & z <
%0.3); view 120 thresh 40
%_5 x=(x > -1.1 & x < 0.4); y=(y > 1.2 & y < 2.3); z= (z > -0.4 & z < 0.3);
%view 0, thresh 40

x=(x > -2.8 & x < -2.3);  
y= (y > -0.5 & y < 0.9); 
z= (z > -0.4 & z < 0.3);


r = x.*y.*z;

intensity = intensity.*r;
pc(:,:,1) = pc(:,:,1).*r;
pc(:,:,2) = pc(:,:,2).*r;
pc(:,:,3) = pc(:,:,3).*r;

intensity(intensity > intensity_factor) = 2000;
intensity(intensity <= intensity_factor) = 0;

x = pc(:,:,1);
y = pc(:,:,2);
z = pc(:,:,3);

x1 = double(x(x > 0 | x < 0));
y1 = double(y(y > 0 | y < 0));
z1 = double(z(z > 0 | z < 0));

figure
pcshow(pc,intensity)
view([90,0])


%%


%bags = {'high', 'medium', 'low', 'low2', 'low3', 'low4', 'low5'};
%bags = {'1000', '800', '700', '600', '500', '400', '300', '250', '950', '150'};
%bags = {'1000', '900', '800', '700', '600', '500', '400', '300', '200', '100'};
%bags = {'1000', '900', '800', '700', '600', '500', '400', '300', '100'};
bags = {'1000_', '900_', '800_', '700_', '600_', '500_', '400_', '300_'};

fileID = fopen('motion/motion_locations/scan.csv','w');

for b=1:5

for f=1:length(bags)

bag = roscpp.bag.internal.RosbagWrapper(strcat('/media/hdd/new_locations/', string(bags(f)), string(b), '.bag'));
out_string = evalc('disp(bag.info)');
expression = 'Start:.* (?<time>\d+:\d+:\d+\.\d+).*End';
tokens1 = regexp(out_string, expression, 'tokens');
expression = 'End:.* (?<time>\d+:\d+:\d+\.\d+).*Size';
tokens2 = regexp(out_string, expression, 'tokens');

fprintf(fileID, "%s,%s\n",  string(tokens1{1}), string(tokens2{1}));


end
end
%%

folder = 'motion_dist_dataset/town_2/frame/';

SortedFilenames = sortComplex(folder, 'data');
SortedStates = sortComplex(folder, 'state');


player = pcplayer([-50 50],[-50 50],[-10 10]);
max_vel = 10;
%while isOpen(player) 
    for i=1:numel(SortedFilenames)
         
         %{
         state = readmatrix(strcat(folder,SortedStates{i}));
         if state(4) < 1
             Msgs = readmatrix(strcat(folder,SortedFilenames{i}));
             pc = reshape(Msgs,[],2000,7);
             velos = pc(:,:,6);
             obj_ids = pc(:,:,4);
             vels = velos(velos > 1);
             
             if max(vels) >= max_vel
                 %idxs = find(velos >= 20);
                 %unique(obj_ids(idxs))
                 xyz = ones(size(pc,1),size(pc,2),3).*(velos >= max_vel);
                 pc = pc(:,:,1:3).*xyz;
                 pcshow(pc)
                 input('continue>')
             end
             
         end
        %}
         Msgs = readmatrix(strcat(folder,SortedFilenames{i}));
         pc = reshape(Msgs,[],2000,7);
         xyz = ones(size(pc,1),size(pc,2),3).*(pc(:,:,4) >= max_vel);
         ptCloud = pointCloud(pc(:,:,1:3).*xyz);
         view(player,ptCloud);    
         %pcshow(pc,intensity)
    end 
%end

function sorted = sortComplex(folder, string_scan)

dd = dir(strcat(folder,string_scan,'_*.csv'));
fileNames = {dd.name};

filenum = cellfun(@(x)sscanf(x,strcat(string_scan,'_%d.csv')), fileNames);
% sort them, and get the sorting order
[~,Sidx] = sort(filenum); 
% use to this sorting order to sort the filenames
sorted = fileNames(Sidx);

end