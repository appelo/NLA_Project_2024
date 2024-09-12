a = dir('u0*.txt');
b = dir('v0*.txt');

u = load(a(1).name);   
n = length(u(:,1))-1;
h = 1/n;

[x,y] = meshgrid((0:h:1));

for i = 1:length(a)
    u = load(a(i).name);   
    v = load(b(i).name);
    w = -(v(2:end-1,3:end)-v(2:end-1,1:end-2))/(2*h) ...
        + (u(3:end,2:end-1)-u(1:end-2,2:end-1))/(2*h);
    clf
    [c,hdl] = contour(x(2:end-1,2:end-1),y(2:end-1,2:end-1),w,[-6:6],'k','linewidth',2);
    clabel(c,hdl);
    %shading interp
    %view(0,90)
    %contour(x(2:end-1,2:end-1),y(2:end-1,2:end-1),w,linspace(-10,10,100))
    %colorbar
    %axis equal 
    %  contour(x,y,sqrt(u.^2+v.^2),linspace(-1,1,30))
    %shading interp
    %    hold on
    %quiver(x,y,u,v,10)
    colorbar
    axis([0 1 0 1])
    axis equal
    drawnow
    print(['frame' num2str(i,'%05d') '.jpg'],'-djpeg')
end
    