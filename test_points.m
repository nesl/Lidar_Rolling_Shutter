resolution = 2048;
num_frames = 2;
fs_scan = 10;
fs = resolution*fs_scan;
frequency = 233;
distance = 3;
angle = 360/resolution;
angle_err = 0.01;
surface_length_h = 2;
surface_length_v = 1;
resolution_h = tand(angle)*distance;
max_length = floor(surface_length_h/resolution_h);
amplitude = 0.001;% 0.001;
angle_surface = 50;
velocity_km_h = 80; %km/h
velocity = velocity_km_h/3600*1000;
initial_position = 2;

distance_to_origin_plane = surface_length_h*sind(angle_surface)+distance;
distance_to_y_axis = -surface_length_h*cosd(angle_surface);
vector = [distance_to_y_axis; (distance_to_origin_plane - distance)];
u_vector = sqrt(vector(1)^2+vector(2)^2);
vector_u = vector/u_vector;
normal_vector = [0 1; -1 0]*vector;
u = sqrt(normal_vector(1)^2+normal_vector(2)^2);
normal_vector_u = normal_vector/u;
vibration_direction = normal_vector_u*amplitude;


points = [[distance_to_y_axis,0];[distance_to_origin_plane,distance]] - [vector(1)/2;vector(2)/2];
side2 = normal_vector_u*surface_length_v+[points(1,2); points(2,2)];
points2 = [[points(1,2),side2(1)];[points(2,2),side2(2)]];
%distance_to_origin_plane = distance_to_origin_plane + vibration_direction(2);
%distance_to_y_axis = distance_to_y_axis + vibration_direction(1);
side3 = vector + [side2(1);side2(2)];
points3 = [[side2(1),side3(1)];[side2(2),side3(2)]];
side4 = normal_vector_u*surface_length_v+[points(1,1); points(2,1)];
points4 = [[side4(1),points(1,1)];[side4(2),points(2,1)]];

%points = points + vibration_direction - [vector(1)/2;vector(2)/2];
%plot([distance_to_y_axis,0+vibration_direction(1)], [distance_to_origin_plane, distance+vibration_direction(2)])

square1 = [points(1,:);points2(1,:);points3(1,:);points4(1,:)] +vector_u(1)*initial_position;
square2 = [points(2,:);points2(2,:);points3(2,:);points4(2,:)] +vector_u(2)*initial_position;




init = 0;
max_distance = 6;
actual_distance = max_distance;
ang = 180;
intersect_f = 0;
period_t = 1/fs;
animate = 0;
points_contact = [];
frame = zeros(resolution*num_frames,1) + max_distance;

if animate
    plot(square1,square2 , 'g')
    hold on
    axis([-max_distance max_distance -max_distance max_distance])
end

for p=1:resolution*num_frames
    
    [x,y] = pol2cart(deg2rad(ang),max_distance);
    %[intersect_f, point_int] = get_line_intersection(points(:,1),points(:,2),[0,0],[x,y]);
    [xi,yi] = polyxpoly(square1,square2, [0,x],[0,y]);
    if xi
        for xii=1:length(xi)
            actual_distance_tmp = sqrt(xi(xii)^2+yi(xii)^2);
            if actual_distance_tmp < actual_distance
                actual_distance = actual_distance_tmp;
            end
        end
    end
    [x,y] = pol2cart(deg2rad(ang),actual_distance);
    if actual_distance < max_distance
        points_contact = cat(1,points_contact,[x,y]);
        frame(p) = actual_distance;
    end
    ang = ang - angle;
    actual_distance = max_distance;
    square1 = square1-vector_u(1)*velocity*period_t;
    square2 = square2-vector_u(2)*velocity*period_t;
    
    
    if animate
        %set(gcf,'Visible','off') 
        plot([0,x],[0,y], 'ro-')
        hold on
        if ~isempty(points_contact)
            plot(points_contact(:,1),points_contact(:,2), 'ro')
        end
        plot(square1,square2 , 'g')
        axis([-max_distance max_distance -max_distance max_distance])
        hold off
        %{
        frame = getframe(gcf);
        img = frame2im(frame);
        [img,cmap] = rgb2ind(img,256);
        if p == 1
            imwrite(img,cmap,'animation.gif','gif','LoopCount',Inf,'DelayTime',0, 'DisposalMethod', 'leaveInPlace');
        else
            imwrite(img,cmap,'animation.gif','gif','WriteMode','append','DelayTime',0, 'DisposalMethod', 'leaveInPlace');
        end
        %}
        pause(0.0001);
    end
end

plot(points_contact(:,1),points_contact(:,2), 'ro')
axis([-max_distance max_distance -max_distance max_distance])
frame = reshape(frame,resolution,num_frames);
i1 = find(frame(:,1) < max_distance);
i2 = find(frame(:,2) < max_distance);

[x1,y1] = pol2cart(deg2rad(180-i1(1)*360/resolution),frame(i1(1),1));
[x2,y2] = pol2cart(deg2rad(180-i2(1)*360/resolution),frame(i2(1),2));
%points_contact = cat(1,points_contact,[x,y]);
vec_dir = [x2,y2]-[x1,y1];
vec_dir_u = vec_dir/sqrt(sum(vec_dir.^2));
CosTheta = max(min(dot(vec_dir_u,[1,0])/(norm(vec_dir_u)*norm([1,0])),1),-1);
ThetaInDegrees =  - real(acosd(CosTheta));

max_stretch = 0;
for cc=1:length(i1)
    [xt,yt] = pol2cart(deg2rad(180-i1(cc)*360/resolution),frame(i1(cc),1));
    realx = global2localcoord([xt;yt;0],"rr",[x1;y1;0], rotz(ThetaInDegrees));
    if realx(1) > max_stretch
        max_stretch = realx(1);
    end
end

max_stretch2 = 0;
for cc=1:length(i2)
    [xt,yt] = pol2cart(deg2rad(180-i2(cc)*360/resolution),frame(i2(cc),2));
    realx = global2localcoord([xt;yt;0],"rr",[x2;y2;0], rotz(ThetaInDegrees));
    if realx(1) > max_stretch2
        max_stretch2 = realx(1);
    end
end
%R = [cosd(ThetaInDegrees),-sind(ThetaInDegrees),x1;sind(ThetaInDegrees),cosd(ThetaInDegrees),y1;0,0,1];

t_angle = (i2(1)-i1(1))*360/resolution;
a = frame(i1(1),1);
b = frame(i2(1),2);
t_distance = sqrt(a^2 + b^2 - 2*a*b*cosd(t_angle));
t_time = (i2(1)+resolution-i1(1))/fs;
t_vel = t_distance/t_time; %*3600/1000
d_frame = length(i1)/fs*t_vel;
car_size = surface_length_h; %((max_stretch - d_frame) + (max_stretch2 - d_frame))/2

mid_p = local2globalcoord([max_stretch;0;0],"rr",[x1;y1;0],rotz(ThetaInDegrees));
vv = mid_p - [x1;y1;0];
a = sqrt(sum(vv.^2));
b = frame(i1(1),1);
[~,c] = cart2pol(mid_p(1),mid_p(2));
alpha = acosd((b^2+c^2-a^2)/(2*b*c));
m_time = alpha/(fs_scan*360);

vel2 = (max_stretch - car_size)/m_time*3600/1000

%new_p = car_size/(frame(i1(1),1)*sind(360/resolution))
%extra_points = 


%%
%equation

speed = 50;
distance = 5;
car_size = 2;
angle_laser = 0.18;
car_angle = 0;
fs_scan = 10;
resolution = 2000;
fs = fs_scan*resolution;

%extra_time = (car_size/(sind(angle_laser)*distance))/fs;
extra_time = (car_size*cosd(car_angle)/(sind(angle_laser)*distance))/fs;
extra_distance = speed*extra_time*cosd(car_angle);
extra_distance/(sind(angle_laser)*distance)

%%

blades = 5;
blade_ang = 360/blades;
blade_sz = 0.5;
blade_ang2 = 0;
num_frames = 0.5;
resolution = 2048;
fs_scan = 10;
fs = resolution*fs_scan;
rpm = 1000;
ang_vel = 360*rpm/(60*fs);
laser_channels = 32;
v_range = 45;
v_angles = linspace(-v_range/2,v_range/2,laser_channels);
distance = 1.5;

angle = 360/resolution;
view_distance = distance*sind(v_angles(length(v_angles)));

view_coords = [-view_distance,view_distance; distance, distance];

max_distance = 6;
actual_distance = ones(1,laser_channels)*max_distance;
distance_vector = ones(1,laser_channels)*distance;

%// radius
r = 0.1;

%triang = (180-blade_ang)/2;

%triside = (r*sind(blade_ang))/sind(triang);
%arc_len = 2*pi*r*(blade_ang/360);




%// center
c = [0 0];

%circ = [c-r 2*r 2*r];
circ = nsidedpoly(1000, 'Center', c, 'Radius', r);

pgon = [[0 r*cosd(blade_ang) blade_sz*cosd(blade_ang2) r]; [0 r*sind(blade_ang) blade_sz*sind(blade_ang2) 0]; [0 0 0 0]];
%rectangle('Position',circ,'Curvature',[1 1])

figp = polyshape(pgon(1,:),pgon(2,:));

%plot(figp)

for i=1:blades-1
    pgon = rotz(blade_ang)*pgon;
    figp2 = polyshape(pgon(1,:),pgon(2,:));
    figp = union(figp,figp2);
end

plot(figp)
hold on
plot(circ)
axis([-view_distance view_distance -view_distance view_distance])
hold off

figp = rotate(figp,10); %initial rotation

tiledlayout('flow')
for rpm = 100:100:2000
    points_contact = [];
    ang = 180;
    ang_vel = 360*rpm/(60*fs);
for p=1:resolution*num_frames
    
    [x,y] = pol2cart(deg2rad(ang),max_distance);
    ang = ang - angle;

    %[intersect_f, point_int] = get_line_intersection(points(:,1),points(:,2),[0,0],[x,y]);
    [xi,yi] = polyxpoly(view_coords(1,:),view_coords(2,:), [0,x],[0,y]);
    
    figp = rotate(figp,-ang_vel);
    %plot(figp)
    %hold on
    %plot(circ)
    
    if xi
        h_pos = ones(1,laser_channels)*xi;
        isinside = figp.isinterior(h_pos,distance_vector.*sind(v_angles)).';
        isinside_circ = circ.isinterior(h_pos,distance_vector.*sind(v_angles)).';
        isinside = isinside | isinside_circ;
        real_points = (max_distance*(~isinside)+distance_vector.*isinside).*sind(v_angles);
        %plot(h_pos,real_points, 'ro')
        if sum(isinside) > 0
            idxs = find(isinside > 0);
            points_contact = cat(1,points_contact,[ones(length(idxs),1)*xi,distance*sind(v_angles(idxs)).']);
        end
    end
    if ~isempty(points_contact)
        %plot(points_contact(:,1),points_contact(:,2), 'ro')
    end
    %axis([-view_distance view_distance -view_distance view_distance])
    %hold off
    %{
    frame = getframe(gcf);
    img = frame2im(frame);
    [img,cmap] = rgb2ind(img,256);
    if p == 1
        imwrite(img,cmap,'animation3.gif','gif','LoopCount',Inf,'DelayTime',0, 'DisposalMethod', 'leaveInPlace');
    else
        imwrite(img,cmap,'animation3.gif','gif','WriteMode','append','DelayTime',0, 'DisposalMethod', 'leaveInPlace');
    end
    %}
    %pause(0.001)
end
nexttile
plot(points_contact(:,1),points_contact(:,2), 'ro')
title(strcat(string(rpm), " RPM"))
end

%%
%45 cm, 0.5 cm vibration asind(0.5/45)

lambda = 865e-9;
beam_diameter = 9.5e-3;
beam_divergence = 0.18; %fwhm
distance = 1;
spot_size = beam_diameter + 2*sind(beam_divergence)*distance;
speckle_size = distance*lambda/(2*spot_size);
vibration_slope = 0.0001;%0.6;
vibr_freq = 500;%5000;
f1 = 0.8;
f2 = -0.1;
max_displacement = vibration_slope*distance;%*f2/f1;
fs = 20480;
factor = 1e6;
aperture = 50;
vertical_resolution = 45/32;
vertical_laser = 2;
horizontal_laser = 2;


v_point = distance*sind(vertical_resolution);
%h_point = 

nx = 5000;%10000;
ny = 1000;%1000;



speckles = model_speckles([nx,ny],round(speckle_size*factor*2));
speckles_anim = speckles;

intensity = [];
counter = 1;


divisions = [1 3];
for i=1:horizontal_laser
    for j=1:vertical_laser
        intensity(counter) = mean(speckles(round(divisions(j)*nx/4)-(aperture/2):round(divisions(j)*nx/4)+(aperture/2),round(divisions(i)*ny/4)-(aperture/2):round(divisions(i)*ny/4)+(aperture/2)),'all');
        counter = counter + 1;
    end
end


%intensity1 = [];
%intensity2 = [];
%mean_intensity1 = mean(speckles(round(nx/4)-(aperture/2):round(nx/4)+(aperture/2),round(ny/4)-(aperture/2):round(ny/4)+(aperture/2)),'all');
%mean_intensity2 = mean(speckles(round(3*nx/4)-(aperture/2):round(3*nx/4)+(aperture/2),round(3*ny/4)-(aperture/2):round(3*ny/4)+(aperture/2)),'all');

%intensity1 = [intensity1 mean_intensity1];
%intensity2 = [intensity2 mean_intensity2];

for fi=1:1000
   imshow(speckles,[0,0.001])
   colormap('hot')
   displacement = max_displacement*sin(2*pi*vibr_freq*fi/fs);
   displacement_um = round(displacement*factor);
   speckles = circshift(speckles,displacement_um,1);
   speckles_anim = cat(4,speckles_anim,speckles);
   
   %{
   mean_intensity1 = mean(speckles(round(nx/4)-(aperture/2):round(nx/4)+(aperture/2),round(ny/4)-(aperture/2):round(ny/4)+(aperture/2)),'all');
   mean_intensity2 = mean(speckles(round(3*nx/4)-(aperture/2):round(3*nx/4)+(aperture/2),round(3*ny/4)-(aperture/2):round(3*ny/4)+(aperture/2)),'all');

   intensity1 = [intensity1 mean_intensity1];
   intensity2 = [intensity2 mean_intensity2];
   %}
   A=[];
   counter = 1;
   hold on
   for i=1:horizontal_laser
     for j=1:vertical_laser
        A(counter) = mean(speckles(round(divisions(j)*nx/4)-(aperture/2):round(divisions(j)*nx/4)+(aperture/2),round(divisions(i)*ny/4)-(aperture/2):round(divisions(i)*ny/4)+(aperture/2)),'all');
        counter = counter + 1;
        rectangle('Position',[round(divisions(i)*ny/4)-(aperture/2),round(divisions(j)*nx/4)-(aperture/2),aperture,aperture], 'EdgeColor','y', 'LineWidth',1)
     end
   end
   intensity = cat(1,intensity,A);
   hold off
    
   %{
   frame = getframe(gcf);
    img = frame2im(frame);
    [img,cmap] = rgb2ind(img,256);
    if fi == 1
        imwrite(img,cmap,'animation2.gif','gif','LoopCount',Inf,'DelayTime',0, 'DisposalMethod', 'leaveInPlace');
    else
        imwrite(img,cmap,'animation2.gif','gif','WriteMode','append','DelayTime',0, 'DisposalMethod', 'leaveInPlace');
    end
   %}
   %pause(0.01);
   %input('input')

end
%%
lambda = 865e-9;
beam_diameter = 9.5e-3;
beam_divergence = 0.18; %fwhm
distance = 1;
spot_size = beam_diameter + 2*sind(beam_divergence)*distance;
speckle_size = distance*lambda/(2*spot_size);
vibration_slope = 0.0001;%0.6;
vibr_freq = 233.5;%230;
f1 = 0.8;
f2 = -0.1;
max_displacement = vibration_slope*distance;%*f2/f1;
points = 2048;
fs_frame = 10;
fs = points*fs_frame;
num_frames = 100;
iterations = num_frames*points;
factor = 1e6;
aperture = 50;
vertical_resolution = 45/32;
vertical_laser = 2;
horizontal_laser = 80;


v_point = distance*sind(vertical_resolution);
%h_point = 

nx = 5000;%10000;
ny = 100;%1000;

speckles = model_speckles([nx,ny],round(speckle_size*factor));
for i=1:horizontal_laser
    speckles = cat(3,speckles, model_speckles([nx,ny],round(speckle_size*factor)));
    
end

intensity = [];
A = [];
done = 1;

%intensity1 = [];
%intensity2 = [];
%mean_intensity1 = mean(speckles(round(nx/4)-(aperture/2):round(nx/4)+(aperture/2),round(ny/4)-(aperture/2):round(ny/4)+(aperture/2)),'all');
%mean_intensity2 = mean(speckles(round(3*nx/4)-(aperture/2):round(3*nx/4)+(aperture/2),round(3*ny/4)-(aperture/2):round(3*ny/4)+(aperture/2)),'all');

intensity1 = [];
mean_intensity1 = mean(speckles(round(nx/2)-(aperture/2):round(nx/2)+(aperture/2),round(ny/2)-(aperture/2):round(ny/2)+(aperture/2),1),'all');
intensity1 = [intensity1 mean_intensity1];
%intensity2 = [intensity2 mean_intensity2];
displacement_um = 0;
for i=1:iterations
   %imshow(speckles,[0,0.001])
   displacement = max_displacement*sin(2*pi*vibr_freq*i/fs);
   displacement_um = displacement_um + round(displacement*factor);

   
   %if i <= 80
   %    intensity(i) = mean(speckles(round(nx/2)-(aperture/2):round(nx/2)+(aperture/2),round(ny/2)-(aperture/2):round(ny/2)+(aperture/2),i),'all');
   if mod(i,2048) <= 80 && mod(i,2048) > 0 
       speckles = circshift(speckles,displacement_um,1);
       A(mod(i,2048)) = mean(speckles(round(nx/2)-(aperture/2):round(nx/2)+(aperture/2),round(ny/2)-(aperture/2):round(ny/2)+(aperture/2),mod(i,2048)),'all');
       done = 1;
       displacement_um = 0;
   elseif done
       intensity = cat(1,intensity,A);
       A = [];
       done = 0;
   end
    %mean_intensity1 = mean(speckles(round(nx/2)-(aperture/2):round(nx/2)+(aperture/2),round(ny/2)-(aperture/2):round(ny/2)+(aperture/2),1),'all');
    %intensity1 = [intensity1 mean_intensity1];

    %{
   mean_intensity1 = mean(speckles(round(nx/4)-(aperture/2):round(nx/4)+(aperture/2),round(ny/4)-(aperture/2):round(ny/4)+(aperture/2)),'all');
   mean_intensity2 = mean(speckles(round(3*nx/4)-(aperture/2):round(3*nx/4)+(aperture/2),round(3*ny/4)-(aperture/2):round(3*ny/4)+(aperture/2)),'all');

   intensity1 = [intensity1 mean_intensity1];
   intensity2 = [intensity2 mean_intensity2];
   %}
   
   
  
   %pause(0.01);

end

%%

%Diffraction of a single slit:
clear
N=128% row number N and column number N
for i =1:N
for j =1:N
    
    C(i,j) =(-1)^(i+j);% C is to centre the pattern
end
end
C;
W=8%W is the slit width
for i =1:N
for j =1:N
if j>N/2-W/2 && j<=N/2+W/2
tr(i,j) =1;% actual transmittance
else
tr(i,j) =0;
end
end
end
t=C.*tr;
fft =fft2(t);%two-dimensional fast Fourier transform
I=abs(fft).*abs(fft);%intensity
colormap(gray)
subplot(2,2,1)
imagesc(tr)
subplot(2,2,2)
imagesc(I)
subplot(2,2,3)
plot(I(N/2+1,:))
return

%%



%%
%plot([0,vibration_direction(1)], [distance,vibration_direction(2)+distance])
 signal_master = [];
real_signal_master = [];
rval = rand(1,1)*(1-max_length/fs);
%points = linspace(rval,rval+(max_length-1)/fs, max_length);
real_points = linspace(rval,rval+(resolution-1)/fs, resolution);

for i=1:num_frames
    %w = window(@hann, max_length).';
    signal = amplitude*sin(2*pi*frequency*real_points(1:max_length)); %.*w;
    real_signal = amplitude*sin(2*pi*frequency*real_points);
    signal = [signal (zeros(1,resolution-max_length))];
    signal_master = [signal_master signal];
    real_signal_master = [real_signal_master real_signal];
    %points = points + 2048/fs;
    real_points = real_points + resolution/fs;
end
figure
plot(real_signal_master)
hold on
plot(signal_master)


figure
S = signal_master; %- mean(signal_master);
%[pxx,f] = plomb(S,fs, 'psd');
[pxx,f] = periodogram(S,[],[],fs);
plot(f,pxx);
%ev = envelope(pxx, 100, 'peak');
%plot(f,ev)

%%

figure
plot(S)

n = length(S);
X = fft(S);
Y = fftshift(X);
fshift = (-n/2:n/2-1)*(fs/n); % zero-centered frequency range
powershift = abs(Y).^2/n;     % zero-centered power
figure
plot(fshift,powershift)


%S = nanmean(stream2(:,:,55),1);
%S = reshape(nanmean(stream2,1),[],1);

function [b,point] = get_line_intersection(A, B, C, D)

    point = zeros(2,1);
    s1_x = B(1)-A(1);
    s1_y = B(2)-A(2);
    s2_x = D(1)-C(1);
    s2_y = D(2)-C(2);

    s = (-s1_y * (A(1) - C(1)) + s1_x * (A(2) - C(2))) / (-s2_x * s1_y + s1_x * s2_y);
    t = ( s2_x * (A(2) - C(2)) - s2_y * (A(1) - C(1))) / (-s2_x * s1_y + s1_x * s2_y);

    if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
        %Collision detected
        point(1) = A(1) + (t * s1_x);
        point(2) = A(2) + (t * s1_y);
        b =  1;
    else
        b = 0; % No collision
    end

end

function b = ccw(A, B, C)
    b =  (C(2)-A(2)) * (B(1)-A(1)) > (B(2)-A(2)) * (C(1)-A(1));
end


function b = intersect(A,B,C,D)
    b= ccw(A,C,D) ~= ccw(B,C,D) && ccw(A,B,C) ~= ccw(A,B,D);
end
    
%{
function b = intersect(points, points2)

a1 = (points(2,2)-points(2,1)) / (points(1,2) - points(1,1));
b1 = points(2,1) - a1*points(1,1);
a2 = (points2(2,2)-points2(2,1)) / (points2(1,2) - points2(1,1));
b2 = points2(2,1) - a2*points2(1,1);
X = (b1-b2)/(a2-a1);

end
    
%}
